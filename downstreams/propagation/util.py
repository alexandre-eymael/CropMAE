import argparse

def get_args_parser():

    parser = argparse.ArgumentParser('Label Propagation Evaluation on DAVIS, JHMDB, and VIP')

    parser.add_argument('--name', default='CropMAE_label_prop', type=str, help='Name of the experiment')
    parser.add_argument('--epoch', default=20, type=int, help='Epoch of the model')
    parser.add_argument('--checkpoint', default='checkpoints/CropMAE.pth', type=str, help='Path to the model checkpoint')

    parser.add_argument('--backbone', default='vits', type=int, choices=['vits', 'vitb', 'vitl'], help='Size of the model')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch size of the model')

    return parser
