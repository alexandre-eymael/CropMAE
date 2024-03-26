import argparse
import torch
import random

def common_args(parser):
    return parser

def test_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Label Propagation')

    # Datasets
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=0, help='manual seed')

    parser.add_argument('--batch_size', default=5, type=int,
                        help='batch_size')
    parser.add_argument('--temperature', default=0.7, type=float,
                        help='temperature')
    parser.add_argument('--topk', default=7, type=int,
                        help='k for kNN')
    parser.add_argument('--radius', default=20, type=float,
                        help='spatial radius to consider neighbors from')
    parser.add_argument('--video_len', default=20, type=int,
                        help='number of context frames')

    parser.add_argument('--crop_size', default=(480, 880), type=int, nargs='+',
                        help='resizing of test image')

    parser.add_argument('--filelist', type=str)
    parser.add_argument('--save_path', default='./results', type=str)

    # Model Details
    parser.add_argument('--model_type', default='vits16', type=str)

    parser.add_argument('--no_l2', default=False, action='store_true', help='')
    parser.add_argument('--norm_mask', default=False, action='store_true', help='')

    parser.add_argument('--long_mem', default=[0], type=int, nargs='*', help='')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        # print('Using GPU', args.gpu_id)
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    return args




