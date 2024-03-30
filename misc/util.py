import torch
from torchvision.utils import make_grid
import random
import argparse

def create_blank_image(dims, device="cuda"):
    return torch.zeros(dims, device=device, dtype=torch.uint8)

def get_args_parser():

    parser = argparse.ArgumentParser('CropMAE pre-training', add_help=False)

    # Architecture parameters
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--architecture', default='vits', type=str, help='architecture (vits, vitb, vitl)')
    parser.add_argument('--decoder_embed_dim', default=256, type=int, help='decoder embedding dimension')
    parser.add_argument('--decoder_depth', default=4, type=int, help='number of decoder layers')
    parser.add_argument('--decoder_num_heads', default=8, type=int, help='number of decoder heads')
    parser.add_argument('--masking_ratio', default=0.95, type=float, help='Masking ratio')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch size')
    parser.add_argument('--norm_pix_loss', action='store_true', help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    
    # Training parameters
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--max_files', default=None, type=int, help='Max number of files  (randomly) taken from the dataset')
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument("--wandb", default="online", type=str, help="Wandb mode (online/offline)")
    parser.add_argument('--wandb_run_name', default="", type=str, help='Wandb run name')
    parser.add_argument('--save_every_n', default=10, type=int, help='Save checkpoint every n epochs')
    parser.add_argument('--log_per_epoch', default=2, type=int, help='Number of logs per epoch')

    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--betas', default='0.9,0.95', type=str, help='betas for Adam optimizer')

    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr_prop', default=0.0, type=float, help="relative minimum lr")

    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--warmup_epochs_prop', default=0.05, type=float, help='relative length of warmup in epochs')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    
    # Augmentation parameters
    parser.add_argument('--random_area_min_global', type=float, default=0.2, help='random area min')
    parser.add_argument('--random_area_max_global', type=float, default=1.0, help='random area max')
    parser.add_argument('--random_area_min_local', type=float, default=0.2, help='random area min')
    parser.add_argument('--random_area_max_local', type=float, default=1.0, help='random area max')

    parser.add_argument('--use_color_jitter', type=bool, default=False, help='use color jitter')
    parser.add_argument('--use_gaussian_blur', type=bool, default=False, help='use gaussian blur')
    parser.add_argument('--use_elastic_transform', type=bool, default=False, help='use elastic transform')

    parser.add_argument('--random_aspect_ratio_min_global', type=float, default=3/4, help='random aspect ratio min')
    parser.add_argument('--random_aspect_ratio_max_global', type=float, default=4/3, help='random aspect ratio max')
    parser.add_argument('--random_aspect_ratio_min_local', type=float, default=3/4, help='random aspect ratio min')
    parser.add_argument('--random_aspect_ratio_max_local', type=float, default=4/3, help='random aspect ratio max')

    parser.add_argument('--repeated_sampling_factor', default=1, type=int, help='repeated sampling factor')
    parser.add_argument("--crop_strategy", default="GlobalToLocal", type=str, help="Crop strategy")
    parser.add_argument("--interpolation_method", default="bilinear", type=str, help="Interpolation method")
    parser.add_argument("--horizontal_flip_p", default=0.5, type=float, help="Horizontal flip probability")

    # Dataset parameters
    parser.add_argument('--data_path', default='k400_true_valid_files.txt', type=str, help='dataset path')
    parser.add_argument('--output_dir', default='CropMAE/output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='CropMAE/output_dir', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--num_workers', default=8, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Evaluation
    parser.add_argument("--slurm_davis", action="store_true", help="Schedule a DAVIS evaluation job each `save_every_n` epochs with SLURM")

    # Help
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')

    return parser

@torch.no_grad()
def parse_img(image):
    # image is [H, W, 3]
    assert image.shape[2] == 3

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device = image.device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device = image.device)

    return torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).type(torch.uint8)

@torch.no_grad()
def run_one_image_mae(model, img, pred, mask):
    
    img = img[0, :, :, :].unsqueeze(0)
    pred = pred[0, :, :].unsqueeze(0)
    mask = mask[0, :].unsqueeze(0)

    pred = model.unpatchify(pred)
    pred = torch.einsum('nchw->nhwc', pred)

    # visualize the masks
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)
    mask = model.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask)

    img = torch.einsum('nchw->nhwc', img)
    masked_img = img * (1 - mask)

    img = parse_img(img[0]).permute(2, 0, 1)
    masked_img = parse_img(masked_img[0]).permute(2, 0, 1)
    reconstruction = parse_img(pred[0]).permute(2, 0, 1)

    # Make grid
    grid = make_grid([img, masked_img, reconstruction], nrow=3, padding=2, pad_value=255)
    caption = "Original, Masked, Reconstruction"

    return grid, caption

@torch.no_grad()
def run_one_image(model, imgs, masked_preds, masked_masks, n_frames):

    model.eval()
    B = imgs.shape[0]

    # Random batch between 0 and B
    b = random.randint(0, B-1)

    imgs = imgs.chunk(n_frames, dim=1)
    imgs = [img.squeeze(1) for img in imgs]
    unmasked_img, masked_imgs = imgs[0], imgs[1:]

    unmasked_img = unmasked_img[b, :, :, :].unsqueeze(0)
    masked_imgs = [img[b, :, :, :].unsqueeze(0) for img in masked_imgs]
    masked_preds = [pred[b, :, :].unsqueeze(0) for pred in masked_preds]
    masked_masks = [mask[b, :].unsqueeze(0) for mask in masked_masks]

    masked_preds = [model.unpatchify(pred) for pred in masked_preds]
    masked_preds = [torch.einsum('nchw->nhwc', pred) for pred in masked_preds]

    # visualize the masks
    masked_masks = [mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3) for mask in masked_masks]
    masked_masks = [model.unpatchify(mask) for mask in masked_masks]
    masked_masks = [torch.einsum('nchw->nhwc', mask) for mask in masked_masks]

    unmasked_img = torch.einsum('nchw->nhwc', unmasked_img)
    masked_imgs = [torch.einsum('nchw->nhwc', img) for img in masked_imgs]

    # masked image
    masked_imgs_masked = [img * (1 - mask) for img, mask in zip(masked_imgs, masked_masks)]

    unmasked_img = parse_img(unmasked_img[0]).permute(2, 0, 1)
    masked_imgs = [parse_img(img[0]).permute(2, 0, 1) for img in masked_imgs]
    masked_imgs_masked = [parse_img(img[0]).permute(2, 0, 1) for img in masked_imgs_masked]
    masked_imgs_reconstructions = [parse_img(img[0]).permute(2, 0, 1) for img in masked_preds]
    
    # Make grid
    grid = make_grid([
        unmasked_img, *masked_imgs,
        unmasked_img, *masked_imgs_masked,
        create_blank_image(unmasked_img.shape, device=unmasked_img.device), *masked_imgs_reconstructions,
    ], nrow=n_frames, padding=2, pad_value=255)

    caption = "Original, " + ", ".join([f"Frame #{i+1}" for i in range(n_frames-1)]) + "\n"
    caption += "Original, " + ", ".join([f"Masked Frame #{i+1}" for i in range(n_frames-1)]) + "\n"
    caption += "N.A, " + ", ".join([f"Reconstruction #{i+1}" for i in range(n_frames-1)])
    model.train()

    return grid, caption