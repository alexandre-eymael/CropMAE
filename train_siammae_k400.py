from data.SiamMAE_PIPE import SiamMAE_VideoDataset
from misc.WandbLogger import WandbLogger
from models.components.mae.util import misc
import json
from models.components.mae.util.misc import NativeScalerWithGradNormCount as NativeScaler
from models.components.mae.util.lr_sched import adjust_learning_rate
import os
import torch
import random
import torch.backends.cudnn as cudnn
import misc.util as util
from misc.slurm import submit_DAVIS_slurm
import numpy as np
import time
import uuid
from pathlib import Path
from models.SiamMAE import SIAM_MODELS
import timm.optim.optim_factory as optim_factory

def main(args):

    run_id = uuid.uuid4().hex[:8]
    args.wandb_run_name = f"{args.wandb_run_name}-{run_id}"

    misc.init_distributed_mode(args)

    # Device
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    global_rank = misc.get_rank()
    if global_rank == 0:
        print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)

    accum_iter = args.accum_iter
    effective_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    # Compute absolute warmup
    assert 0.0 <= args.warmup_epochs_prop <= 1.0, "warmup_epochs_prop must be in [0, 1]"
    args.warmup_epochs = int(args.warmup_epochs_prop * args.epochs)

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * effective_batch_size / 256

    # Compute absolute min_lr
    assert 0.0 <= args.min_lr_prop <= 1.0, "min_lr_prop must be in [0, 1]"
    args.min_lr = args.min_lr_prop * args.lr

    if global_rank == 0:
        print("Base LR: %.2e" % (args.lr * 256 / effective_batch_size))
        print("Actual LR: %.2e" % args.lr)
        print(f"Effective Batch Size: {effective_batch_size} (= {args.batch_size} * {args.accum_iter} * {misc.get_world_size()})")

    try:
        model = SIAM_MODELS[args.architecture](
            norm_pix_loss=args.norm_pix_loss,
            patch_size=args.patch_size,
            decoder_embed_dim=args.decoder_embed_dim,
            decoder_depth=args.decoder_depth,
            decoder_num_heads=args.decoder_num_heads
        )
    except KeyError:
        raise ValueError(f"Architecture {args.model} not found")

    model = model.to(device)

    # Check if we are in a distributed setting
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Optimizer parameters
    betas = list(map(float, args.betas.split(","))) # convert betas argument to a float tuple
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=betas)

    loss_scaler = NativeScaler()

    # Resume
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Dataset parameters
    dataset_location = args.data_path

    # Read all videos
    video_files = []
    with open(dataset_location, "r") as f:
        for line in f:
            video_files.append(line.strip())

    if global_rank == 0:
        print("Number of videos:", len(video_files))

    # keep only the first `args.max_files` files
    if args.max_files is not None:
        video_files = video_files[:min(args.max_files, len(video_files))]
    args.max_files = len(video_files)

    dataset = SiamMAE_VideoDataset(
        files=video_files,
        args=args
    )

    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(dataset)
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size // args.repeated_sampling_factor, # Scale with repeated sampling factor
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    args.log_every_n = (len(dataloader) // args.repeated_sampling_factor) // args.log_per_epoch
    if args.log_every_n == 0:
        args.log_every_n = (len(dataloader) // args.repeated_sampling_factor)

    # Logger
    args.effective_batch_size = effective_batch_size
    args.optimizer = optimizer.__class__.__name__
    args.dataloader_size = (len(dataloader) // args.repeated_sampling_factor)
    args.__dict__.update(model_without_ddp.n_params)

    if global_rank == 0:
        logger = WandbLogger(
            config=args,
            mode=args.wandb,
            name=args.wandb_run_name if len(args.wandb_run_name) > 0 else None
        )
        # Change args.output_dir to wandb dir
        if args.output_dir is not None:
            if args.wandb == "online":
                args.output_dir = f"{args.output_dir}/{logger.name}"
            else:
                args.output_dir = f"{args.output_dir}/{logger.id}"
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            with open(f"{args.output_dir}/args.json", "w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=4)
        print("{}".format(args).replace(', ', ',\n'))
    else:
        logger = None

    model.train(True)
    optimizer.zero_grad(set_to_none=True)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    for epoch in range(args.start_epoch, args.epochs):

        header = 'Epoch: [{}]'.format(epoch)
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)

        epoch_start = time.time()
        running_loss = 0
        for idx, imgs in enumerate(metric_logger.log_every(dataloader, args.log_every_n, header)):

            if idx >= (len(dataloader) // args.repeated_sampling_factor):
                break

            imgs = imgs.to(device, non_blocking=True)

            # Reshape to take repeated samples into account
            imgs = imgs.view(args.batch_size, 2, 3, args.input_size, args.input_size)

            with torch.cuda.amp.autocast():
                loss, masked_preds, masked_masks = model(
                    imgs,
                    mask_ratio=args.masking_ratio
                )

            loss /= accum_iter

            loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(idx + 1) % accum_iter == 0)

            if (idx + 1) % accum_iter == 0:
                lr = adjust_learning_rate(optimizer, idx / (len(dataloader) // args.repeated_sampling_factor) + epoch, args)
                optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()

            if global_rank == 0:
                running_loss += loss.item()
                if (idx + 1) % args.log_every_n == 0:
                    mean_loss = running_loss / args.log_every_n
                    running_loss = 0
                    imgs_grid, caption = util.run_one_image(model_without_ddp, imgs, masked_preds, masked_masks, 2)
                    logger.log({
                        "epoch" : epoch,
                        "imgs" : logger.make_image(imgs_grid, caption=caption),
                        "loss" :  mean_loss,
                        "lr" : lr,
                        "epoch_progress" : idx / (len(dataloader) // args.repeated_sampling_factor),
                    })

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=lr)

        epoch_end = time.time()
        print(f"Epoch {epoch} took {epoch_end - epoch_start:.2f} seconds")
        if global_rank == 0:
            logger.log({
                "global_progress" : (epoch + 1) / args.epochs,
                "mins_per_epoch" : (epoch_end - epoch_start) / 60,
                "eta_hours" : (args.epochs - (epoch+1)) * (epoch_end - epoch_start) / 3600
            })

        # Save checkpoint
        if args.output_dir and (epoch+1) % (args.save_every_n) == 0:
            save_path = misc.save_model(args=args, epoch=epoch, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
            if global_rank == 0 and args.slurm_davis:
                submit_DAVIS_slurm(logger.get_name(), epoch, save_path)

    last_checkpoint_path = misc.save_model(args=args, epoch=epoch, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if global_rank == 0 and args.slurm_davis:
        submit_DAVIS_slurm(logger.get_name(), epoch, last_checkpoint_path)
        logger.finish()

    torch.distributed.barrier()

    time.sleep(5)
    print(f"Process {global_rank} finished")

if __name__ == "__main__":
    args = util.get_args_parser()
    args = args.parse_args()
    main(args)