import argparse

def get_args_parser():

    parser = argparse.ArgumentParser('Label Propagation Evaluation on DAVIS, JHMDB, and VIP')

    # General arguments
    parser.add_argument('--output_dir', default='downstreams/propagation/cropmae_eval', type=str, help='Output directory')

    # Datasets-related arguments
    parser.add_argument('--davis_path', default='datasets/davis-2017/DAVIS_480_880/', type=str, help='Path to the DAVIS dataset')
    parser.add_argument('--davis_file', default='downstreams/propagation/davis_vallist_480_880.txt', type=str, help='Path to the DAVIS filelist')
    parser.add_argument('--jhmdb_file', default='downstreams/propagation/jhmdb_vallist.txt', type=str, help='Path to the JHMDB filelist')
    parser.add_argument('--vip_file', default='downstreams/propagation/vip_vallist.txt', type=str, help='Path to the VIP filelist')

    # Model-related arguments
    parser.add_argument('--checkpoint', default='checkpoints/CropMAE.pth', type=str, help='Path to the model checkpoint')
    parser.add_argument('--backbone', default='vits', type=str, choices=['vits', 'vitb', 'vitl'], help='Size of the model')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch size of the model')

    return parser
