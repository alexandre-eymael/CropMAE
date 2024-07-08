import os
import csv
from argparse import Namespace
import sys
import torch
import wandb

from .util import get_args_parser as get_prop_args_parser

# Part 1: Label Propagation
from .src.test import main as label_propagation
def _label_propagation(args):

    print("PART 1/3: LABEL PROPAGATION")
    needed_args = set(["model_type", "resume", "save_path", "temperature", "topk", "radius", "video_len", "filelist", "crop_size"])

    # Default arguments
    def_args = Namespace(
        model_type='vits16',
        resume='',
        save_path='./results/',
        temperature=0.7,
        topk=7,
        radius=20,
        video_len=20,
        crop_size=[480, 880],
        filelist='',
        batch_size=5,
        workers=4,
        seed=0,
        no_l2=False,
        norm_mask=False,
        long_mem=[0],
    )

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        def_args.device = 'cuda'
    else:
        def_args.device = 'cpu'

    for arg in needed_args:
        if arg not in args:
            raise ValueError(f"Missing argument: {arg}")
        elif arg not in def_args:
            raise ValueError(f"Unknown argument: {arg}")
        else:
            setattr(def_args, arg, args[arg])

    def_args.img_size = def_args.crop_size

    print(f"Label propagation arguments: {def_args}")
    return label_propagation(def_args)

# Part 2: Conversion
from .src.eval.convert_davis import main as convert_davis
def _convert_davis(args):

    print("PART 2/3: CONVERSION")
    needed_args = set(["out_folder", "in_folder", "dataset", "set"])

    def_args = Namespace(
        out_folder='',
        in_folder='',
        dataset='',
        set=''
    )

    for arg in needed_args:
        if arg not in args:
            raise ValueError(f"Missing argument: {arg}")
        elif arg not in def_args:
            raise ValueError(f"Unknown argument: {arg}")
        else:
            setattr(def_args, arg, args[arg])
    
    print(f"Davis Conversion arguments: {args}")
    return convert_davis(def_args)

# Part 3: Evaluation
from .davis_evaluation.evaluation_method import main as evaluation_davis
from .davis_evaluation.evaluation_method import parse_args as parse_evaluation_davis_args
def _evaluation_davis(args):
    
    print("PART 3/3: EVALUATION")
    needed_args = set(["davis_path", "set", "task", "results_path"])

    # Define the default values for the arguments
    default_davis_path = '/path/to/default_davis_folder'
    default_set = 'val'
    default_task = 'unsupervised'
    default_results_path = ''

    # Create a Namespace object and set its attributes
    def_args = Namespace(
        davis_path=default_davis_path,
        set=default_set,
        task=default_task,
        results_path=default_results_path
    )

    for arg in needed_args:
        if arg not in args:
            raise ValueError(f"Missing argument: {arg}")
        elif arg not in def_args:
            raise ValueError(f"Unknown argument: {arg}")
        else:
            setattr(def_args, arg, args[arg])
    
    print(f"Davis Evaluation arguments: {args}")
    return evaluation_davis(def_args)

from .src.eval.eval_pck import main as evaluation_jhmdb
def _evaluation_jhmdb(args):
    print("PART 3/3: EVALUATION")
    needed_args = set(["filelist", "src_folder", "feat_res"])

    def_args = Namespace(
        filelist='jhmdb_vallist.txt',
        src_folder='',
        feat_res=[20, 20]
    )

    for arg in needed_args:
        if arg not in args:
            raise ValueError(f"Missing argument: {arg}")
        elif arg not in def_args:
            raise ValueError(f"Unknown argument: {arg}")
        else:
            setattr(def_args, arg, args[arg])
    
    print(f"JHMDB Evaluation arguments: {args}")
    return evaluation_jhmdb(def_args)

from .src.eval.eval_vip import main as evaluation_vip
def _evaluation_vip(args):
    print("PART 3/3: EVALUATION")
    needed_args = set(["gt_dir", "pre_dir"])

    def_args = Namespace(
        gt_dir='datasets/VIP/Annotations/Category_ids/',
        pre_dir=''
    )

    for arg in needed_args:
        if arg not in args:
            raise ValueError(f"Missing argument: {arg}")
        elif arg not in def_args:
            raise ValueError(f"Unknown argument: {arg}")
        else:
            setattr(def_args, arg, args[arg])
    
    print(f"VIP Evaluation arguments: {args}")
    return evaluation_vip(def_args)

def evaluate_on_davis(label_prop_args, conversion_davis_args, evaluation_davis_args):
    """
    Perform evaluation on davis and parse the result file.
    The result file is a CSV and is parsed as follows (example):
    ```
    J&F-Mean,J-Mean,J-Recall,J-Decay,F-Mean,F-Recall,F-Decay
    0.506,0.478,0.540,0.220,0.533,0.619,0.270
    ```
    """
    # Perform evaluation on DAVIS
    
    print(
        f"Performing evaluation on DAVIS with the following arguments:\n"
        f"Label Propagation: {label_prop_args}\n"
        f"Conversion: {conversion_davis_args}\n"
        f"Evaluation: {evaluation_davis_args}"
    )

    _label_propagation(label_prop_args)
    _convert_davis(conversion_davis_args)
    res_file = _evaluation_davis(evaluation_davis_args)
    metrics = set(["J&F-Mean", "J-Mean", "J-Recall", "J-Decay", "F-Mean", "F-Recall", "F-Decay"])
    # Open results file as CSV
    with open(res_file, "r") as f:
        reader = csv.reader(f, delimiter=",")
        header = next(reader)
        if set(header) != metrics:
            raise ValueError(f"Invalid header: {header}")
        # Map metric to value
        res = next(reader)
        res = {metric: float(val) for metric, val in zip(header, res)}
    print(f"Results: {res}")
    return res


def evaluate_on_jhmdb(label_prop_args, evaluation_davis_args):
    
    print(
        f"Performing evaluation on JHMDB with the following arguments:\n"
        f"Label Propagation: {label_prop_args}\n"
        f"Evaluation: {evaluation_davis_args}"
    )

    _label_propagation(label_prop_args)
    PCK = _evaluation_jhmdb(evaluation_davis_args)
    PCK_dict = {}
    for k, v in PCK.items():
        print(f"PCK_{k}: {v[0]}")
        PCK_dict[f"PCK_{k}"] = v[0]
    return PCK_dict

def evaluate_on_vip(label_prop_args, evaluation_davis_args):

    print(
        f"Performing evaluation on VIP with the following arguments:\n"
        f"Label Propagation: {label_prop_args}\n"
        f"Evaluation: {evaluation_davis_args}"
    )

    _label_propagation(label_prop_args)
    res = _evaluation_vip(evaluation_davis_args)
    print(f"mIoU: {res}")
    vip_dict = {"mIoU": res}
    return vip_dict
    
if __name__ == "__main__":

    # Arguments
    args = get_prop_args_parser().parse_args()
    model_type = f"{args.backbone}{args.patch_size}"

    davis_prop_args = {
        #"davis_root" : DAVIS_ROOT,
        "model_type": model_type,
        "resume": args.checkpoint,
        "save_path": f"{args.output_dir}/davis/in/",
        "temperature": args.davis_temperature,
        "topk": args.davis_topk,
        "radius": args.davis_radius,
        "video_len": args.davis_video_len,
        "crop_size": args.davis_crop_size,
        "filelist": args.davis_file
    }
    conversion_davis_args = {
        "out_folder": f"{args.output_dir}/davis/out/",
        "in_folder": f"{args.output_dir}/davis/in/",
        "dataset": args.davis_path,
        "set" : "val"
    }
    evaluation_davis_args = {
        "davis_path": args.davis_path,
        "set": "val",
        "task": "semi-supervised",
        "results_path": f"{args.output_dir}/davis/out/"
    }

    jhmdb_prop_args = {
        "model_type": model_type,
        "resume": args.checkpoint,
        "save_path": f"{args.output_dir}/jhmdb/",
        "temperature": args.jhmdb_temperature,
        "topk": args.jhmdb_topk,
        "radius": args.jhmdb_radius,
        "video_len": args.jhmdb_video_len,
        "crop_size": args.jhmdb_crop_size,
        "filelist": args.jhmdb_file
    }

    evaluation_jhmdb_args = {
        "filelist": args.jhmdb_file,
        "src_folder": f"{args.output_dir}/jhmdb/",
        "feat_res": args.jhmdb_feat_res
    }

    vip_prop_args = {
        "model_type": model_type,
        "resume": args.checkpoint,
        "save_path": f"{args.output_dir}/vip/",
        "temperature": args.vip_temperature,
        "topk": args.vip_topk,
        "radius": args.vip_radius,
        "video_len": args.vip_video_len,
        "crop_size": args.vip_crop_size,
        "filelist": args.vip_file
    }

    evaluation_vip_args = {
        "gt_dir": f"{args.vip_path}/Annotations/Category_ids/",
        "pre_dir": f"{args.output_dir}/vip/"
    }

    os.environ['WANDB_DISABLE_SERVICE'] = "True"
    wandb.init(
        mode="online" if args.wandb else "disabled",
        config = {
            "output_dir": args.output_dir,
            "checkpoint": args.checkpoint,
            "params" : {
                "davis_prop_args": davis_prop_args,
                "conversion_davis_args": conversion_davis_args,
                "evaluation_davis_args": evaluation_davis_args,
                "jhmdb_prop_args": jhmdb_prop_args,
                "evaluation_jhmdb_args": evaluation_jhmdb_args,
                "vip_prop_args": vip_prop_args,
                "evaluation_vip_args": evaluation_vip_args
            }
        }
    )

    if args.davis:
        res_davis = evaluate_on_davis(davis_prop_args, conversion_davis_args, evaluation_davis_args)
        print("DAVIS RESULTS:", res_davis)
        wandb.log(res_davis)
    if args.jhmdb:
        res_jhmdb = evaluate_on_jhmdb(jhmdb_prop_args, evaluation_jhmdb_args)
        print("JHMDB RESULTS:", res_jhmdb)
        wandb.log(res_jhmdb)
    if args.vip:
        res_vip = evaluate_on_vip(vip_prop_args, evaluation_vip_args)
        print("VIP RESULTS:", res_vip)
        wandb.log(res_vip)