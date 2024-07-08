from simple_slurm import Slurm
import datetime

def submit_DAVIS_slurm(name, epoch, checkpoint, wandb=True, eval_davis=True, eval_jhmdb=False, eval_vip=False, patch_size=16, backbone="vits"):

    out_dir = f"downstreams/propagation/{name}_{epoch}"

    # header
    slurm = Slurm(
        time=datetime.timedelta(hours=2),
        job_name="DAVIS",
        partition="gpu",
        nodes=1,
        output=f"./logs/{Slurm.JOB_ID}_{Slurm.JOB_NAME}.out",
        gpus_per_node=1,
        cpus_per_gpu=8,
        mem_per_gpu="50G"
    )

    # modules
    slurm.add_cmd("module purge")
    slurm.add_cmd("module load EasyBuild/2023a")
    slurm.add_cmd("module load CUDA/12.2.0")

    # directory
    slurm.add_cmd("cd CropMAE")

    command = "python3 -m downstreams.propagation.start"
    command += f" --output_dir={out_dir}"
    command += f" --checkpoint={checkpoint}"
    command += f" --backbone={backbone}"
    command += f" --patch_size={patch_size}"
    if wandb:
        command += " --wandb"
    if eval_davis:
        command += " --davis"
    if eval_jhmdb:
        command += " --jhmdb"
    if eval_vip:
        command += " --vip"

    # command
    print(f"Creating a job to evaluate {name}@{epoch} with checkpoint {checkpoint} with command: `{command}`")
    slurm.sbatch(command)

    return slurm