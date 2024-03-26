from simple_slurm import Slurm
import datetime

def submit_DAVIS_slurm(name, epoch, checkpoint):

    print(f"Creating a job to evaluate {name}@{epoch} with checkpoint {checkpoint} on DAVIS...")

    # header
    slurm = Slurm(
        time=datetime.timedelta(hours=2),
        job_name="DAVIS",
        partition="gpu",
        nodes=1,
        output=f"./logs/{Slurm.JOB_ID}_{Slurm.JOB_NAME}.out",
        gpus_per_node=1,
        cpus_per_gpu=8,
        mem_per_gpu="50G",
        account="your-account"
    )

    # modules
    slurm.add_cmd("module purge")
    slurm.add_cmd("module load EasyBuild/2023a")
    slurm.add_cmd("module load CUDA/12.2.0")

    # directory
    slurm.add_cmd("cd CropMAE")

    # command
    slurm.sbatch(f"python3 -m downstreams.segmentation.start {name} {epoch} {checkpoint}")

    return slurm