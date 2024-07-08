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

    name = args.name
    epoch = int(args.epoch)
    checkpoint = args.checkpoint

    VIT_PATCH_SIZE = int(args.patch_size)
    backbone = args.backbone

    model_type = f"{backbone}{VIT_PATCH_SIZE}"
    SET = "val" 
    DAVIS_ROOT = "downstreams/propagation"
    
    DAVIS_DATASET = "datasets/davis-2017/DAVIS_480_880/"
    DAVIS_FILE = "downstreams/propagation/davis_vallist_480_880.txt"
    JHMDB_FILE = "downstreams/propagation/jhmdb_vallist.txt"
    VIP_FILE = "downstreams/propagation/vip_vallist.txt"

    davis_prop_args = {
        "davis_root" : DAVIS_ROOT,
        "model_type": model_type,
        "resume": checkpoint,
        "save_path": f"{DAVIS_ROOT}/{name}_{epoch}/in/",
        "temperature": 0.7,
        "topk": 7,
        "radius": 20,
        "video_len": 20,
         "crop_size": [480, 880],
        "filelist": DAVIS_FILE
    }
    conversion_davis_args = {
        "out_folder": f"{DAVIS_ROOT}/{name}_{epoch}/out/",
        "in_folder": f"{DAVIS_ROOT}/{name}_{epoch}/in/",
        "dataset": DAVIS_DATASET,
        "set" : SET
    }
    evaluation_davis_args = {
        "davis_path": DAVIS_DATASET,
        "set": SET,
        "task": "semi-supervised",
        "results_path": f"{DAVIS_ROOT}/{name}_{epoch}/out/"
    }

    jhmdb_prop_args = {
        "model_type": model_type,
        "resume": checkpoint,
        "save_path": f"{DAVIS_ROOT}/{name}_{epoch}/jhmdb/",
        "temperature": 0.7,
        "topk": 7,
        "radius": 20,
        "video_len": 20,
        "crop_size": [320, 320],
        "filelist": JHMDB_FILE
    }

    evaluation_jhmdb_args = {
        "filelist": JHMDB_FILE,
        "src_folder": f"{DAVIS_ROOT}/{name}_{epoch}/jhmdb/",
        "feat_res": (20, 20)
    }

    vip_prop_args = {
        "model_type": model_type,
        "resume": checkpoint,
        "save_path": f"{DAVIS_ROOT}/{name}_{epoch}/vip/",
        "temperature": 0.7,
        "topk": 10,
        "radius": 20,
        "video_len": 20,
        "crop_size": [880],
        "filelist": VIP_FILE
    }

    evaluation_vip_args = {
        "gt_dir": "datasets/VIP/Annotations/Category_ids/",
        "pre_dir": f"{DAVIS_ROOT}/{name}_{epoch}/vip/"
    }

    os.environ['WANDB_DISABLE_SERVICE'] = "True"
    wandb.init(
        mode="online",
        name=f"{name}_{epoch}",
        config = {
            "name": name,
            "epoch": epoch,
            "checkpoint": checkpoint,
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

    res_davis = evaluate_on_davis(davis_prop_args, conversion_davis_args, evaluation_davis_args)
    res_jhmdb = evaluate_on_jhmdb(jhmdb_prop_args, evaluation_jhmdb_args)
    res_vip = evaluate_on_vip(vip_prop_args, evaluation_vip_args)
    
    print("Results:")
    print("DAVIS:", res_davis)
    print("JHMDB", res_jhmdb)
    print("VIP", res_vip)

    wandb.log(res_davis)
    wandb.log(res_jhmdb)
    wandb.log(res_vip)