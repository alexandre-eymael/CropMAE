from __future__ import print_function

import os
import time
import numpy as np

import torch

from .data import vos, jhmdb

from . import utils
from .utils import test_utils as test_utils
from . import models_mae

def main(args):

    if args.model_type == 'vits16':
        model = models_mae.vits16(ckpt_path=args.resume)
        model = model.to(args.device)
        args.map_scale = np.array([16, 16])
    elif args.model_type == 'vits8':
        model = models_mae.vits8(ckpt_path=args.resume)
        model = model.to(args.device)
        args.map_scale = np.array([8, 8])
    elif args.model_type == 'vitb16':
        model = models_mae.vitb16(ckpt_path=args.resume)
        model = model.to(args.device)
        args.map_scale = np.array([16, 16])
    elif args.model_type == 'vitl16':
        model = models_mae.vitl16(ckpt_path=args.resume)
        model = model.to(args.device)
        args.map_scale = np.array([16, 16])
    

    dataset = (vos.VOSDataset if not 'jhmdb' in args.filelist  else jhmdb.JhmdbSet)(args)
    val_loader = torch.utils.data.DataLoader(dataset,
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True) #change worker

    # cudnn.benchmark = False
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    model.eval()
    model = model.to(args.device)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    with torch.no_grad():
        test(val_loader, model, args)
            

def test(loader, model, args):
    n_context = args.video_len
    D = None    # Radius mask

    for vid_idx, (imgs, imgs_orig, lbls, _, lbl_map, meta) in enumerate(loader):
        t_vid = time.time()
        imgs = imgs.to(args.device)
        B, N = imgs.shape[:2]
        h, w = imgs.shape[-2:]
        grid_h, grid_w = h//args.map_scale[0], w//args.map_scale[1]
        assert(B == 1)

        print('******* Vid %s (%s frames) *******' % (vid_idx, N))
        with torch.no_grad():
            t00 = time.time()

            ##################################################################
            # Compute image features (batched for memory efficiency)
            ##################################################################
            bsize = args.batch_size   # minibatch size for computing features
            feats = []
            for b in range(0, imgs.shape[1], bsize):
                if args.model_type == "scratch":
                    feat = model.encoder(imgs[:, b:b+bsize].transpose(1,2).to(args.device))
                elif "vit" in args.model_type:
                    feat_vit = model(imgs[0, b:b+bsize].to(args.device))
                    feat_vit = feat_vit.unsqueeze(0)
                    feat_vit = feat_vit[:, :, 1:]
                    feat_vit = feat_vit.view(feat_vit.shape[0], feat_vit.shape[1], grid_h, grid_w, -1).permute(0, 4, 1, 2, 3)
                    feat = feat_vit
                feats.append(feat.cpu())
            feats = torch.cat(feats, dim=2).squeeze(1)

            if not args.no_l2:
                feats = torch.nn.functional.normalize(feats, dim=1)

            print('computed features', time.time()-t00)

            ##################################################################
            # Compute affinities
            ##################################################################
            torch.cuda.empty_cache()
            t03 = time.time()
            
            # Prepare source (keys) and target (query) frame features
            key_indices = test_utils.context_index_bank(n_context, args.long_mem, N - n_context)
            key_indices = torch.cat(key_indices, dim=-1)           
            keys, query = feats[:, :, key_indices], feats[:, :, n_context:]

            # Make spatial radius mask TODO use torch.sparse
            restrict = utils.MaskedAttention(args.radius, flat=False)
            D = restrict.mask(*feats.shape[-2:])[None]
            D = D.flatten(-4, -3).flatten(-2)
            D[D==0] = -1e10; D[D==1] = 0

            # Flatten source frame features to make context feature set
            keys, query = keys.flatten(-2), query.flatten(-2)

            print('computing affinity')
            Ws, Is = test_utils.mem_efficient_batched_affinity(query, keys, D, 
                        args.temperature, args.topk, args.long_mem, args.device)

            if torch.cuda.is_available():
                print(time.time()-t03, 'affinity forward, max mem', torch.cuda.max_memory_allocated() / (1024**2))

            ##################################################################
            # Propagate Labels and Save Predictions
            ###################################################################

            maps, keypts = [], []
            lbls[0, n_context:] *= 0 
            lbl_map, lbls = lbl_map[0], lbls[0]

            for t in range(key_indices.shape[0]):
                # Soft labels of source nodes
                ctx_lbls = lbls[key_indices[t]].to(args.device)
                ctx_lbls = ctx_lbls.flatten(0, 2).transpose(0, 1)

                # Weighted sum of top-k neighbours (Is is index, Ws is weight) 
                pred = (ctx_lbls[:, Is[t]] * Ws[t].to(args.device)[None]).sum(1)
                pred = pred.view(-1, *feats.shape[-2:])
                pred = pred.permute(1,2,0)
                
                if t > 0:
                    lbls[t + n_context] = pred
                else:
                    pred = lbls[0]
                    lbls[t + n_context] = pred

                if args.norm_mask:
                    pred[:, :, :] -= pred.min(-1)[0][:, :, None]
                    pred[:, :, :] /= pred.max(-1)[0][:, :, None]

                # Save Predictions            
                cur_img = imgs_orig[0, t + n_context].permute(1, 2, 0).numpy() * 255
                _maps = []

                if 'davis' in args.filelist:
                    # os.makedirs(os.path.join(args.save_path, meta["folder_path"][0].split("/")[-1]), exist_ok=True)
                    # outpath = os.path.join(args.save_path, meta["folder_path"][0].split("/")[-1], meta["lbl_paths"][t+n_context][0].split('/')[-1].split('.')[0])
                    outpath = os.path.join(args.save_path, str(vid_idx) + '_' + str(t))
                if 'jhmdb' in args.filelist.lower():
                    coords, pred_sharp = test_utils.process_pose(pred, lbl_map)
                    pose_map = test_utils.vis_pose(np.array(cur_img).copy(), coords.numpy() * args.map_scale[..., None])
                    import imageio
                    imageio.imwrite(os.path.join(args.save_path, str(vid_idx) + '_' + str(t) + '_pose.jpg'), np.uint8(pose_map))
                    _maps += [pose_map]
                    keypts.append(coords)
                    outpath = os.path.join(args.save_path, str(vid_idx) + '_' + str(t))
                if 'vip' in args.filelist:
                    outpath = os.path.join(args.save_path, 'videos'+meta['img_paths'][t+n_context][0].split('videos')[-1])
                    os.makedirs(os.path.dirname(outpath), exist_ok=True)
                    outpath = outpath.replace(".jpg", "")

                heatmap, lblmap, heatmap_prob = test_utils.dump_predictions(
                    pred.cpu().numpy(),
                    lbl_map, cur_img, outpath)

                _maps += [heatmap, lblmap, heatmap_prob]
                maps.append(_maps)

            if len(keypts) > 0:
                coordpath = os.path.join(args.save_path, str(vid_idx) + '.dat')
                np.stack(keypts, axis=-1).dump(coordpath)
                
            torch.cuda.empty_cache()
            print('******* Vid %s TOOK %s *******' % (vid_idx, time.time() - t_vid))


if __name__ == '__main__':
    args = utils.arguments.test_args()

    args.img_size = args.crop_size
    print('Context Length:', args.video_len, 'Image Size:', args.img_size)
    print('Arguments', args)

    main(args)
