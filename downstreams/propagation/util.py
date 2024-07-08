import argparse

def get_args_parser():

    parser = argparse.ArgumentParser('Label Propagation Evaluation on DAVIS, JHMDB, and VIP')

    # General arguments
    parser.add_argument('--output_dir', default='downstreams/propagation/cropmae_eval', type=str, help='Output directory')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--davis', action='store_true', help='Evaluate on DAVIS')
    parser.add_argument('--jhmdb', action='store_true', help='Evaluate on JHMDB')
    parser.add_argument('--vip', action='store_true', help='Evaluate on VIP')

    # Datasets-related arguments
    parser.add_argument('--davis_path', default='datasets/davis-2017/DAVIS_480_880/', type=str, help='Path to the DAVIS dataset')
    parser.add_argument('--davis_file', default='downstreams/propagation/davis_vallist_480_880.txt', type=str, help='Path to the DAVIS filelist')
    parser.add_argument('--jhmdb_path', default='datasets/jhmdb/', type=str, help='Path to the JHMDB dataset')
    parser.add_argument('--jhmdb_file', default='downstreams/propagation/jhmdb_vallist.txt', type=str, help='Path to the JHMDB filelist')
    parser.add_argument('--vip_path', default='datasets/VIP/', type=str, help='Path to the VIP dataset')
    parser.add_argument('--vip_file', default='downstreams/propagation/vip_vallist.txt', type=str, help='Path to the VIP filelist')

    # Model-related arguments
    parser.add_argument('--checkpoint', default='checkpoints/CropMAE.pth', type=str, help='Path to the model checkpoint')
    parser.add_argument('--backbone', default='vits', type=str, choices=['vits', 'vitb', 'vitl'], help='Size of the model')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch size of the model')

    # DAVIS-related evaluation arguments
    parser.add_argument('--davis_temperature', default=0.7, type=float, help='Temperature for the label propagation')
    parser.add_argument('--davis_topk', default=7, type=int, help='Number of top-k predictions to consider')
    parser.add_argument('--davis_radius', default=20, type=int, help='Radius for the label propagation')
    parser.add_argument('--davis_video_len', default=20, type=int, help='Length of the video')
    parser.add_argument('--davis_crop_size', default=[480, 880], type=list, help='Size of the crop')

    # JHMDB-related evaluation arguments
    parser.add_argument('--jhmdb_temperature', default=0.7, type=float, help='Temperature for the label propagation')
    parser.add_argument('--jhmdb_topk', default=7, type=int, help='Number of top-k predictions to consider')
    parser.add_argument('--jhmdb_radius', default=20, type=int, help='Radius for the label propagation')
    parser.add_argument('--jhmdb_video_len', default=20, type=int, help='Length of the video')
    parser.add_argument('--jhmdb_crop_size', default=[320, 320], type=list, help='Size of the crop')
    parser.add_argument('--jhmdb_feat_res', default=[20, 20], type=list, help='Feature resolution')

    # VIP-related evaluation arguments
    parser.add_argument('--vip_temperature', default=0.7, type=float, help='Temperature for the label propagation')
    parser.add_argument('--vip_topk', default=10, type=int, help='Number of top-k predictions to consider')
    parser.add_argument('--vip_radius', default=20, type=int, help='Radius for the label propagation')
    parser.add_argument('--vip_video_len', default=20, type=int, help='Length of the video')
    parser.add_argument('--vip_crop_size', default=[880], type=list, help='Size of the crop')

    return parser
