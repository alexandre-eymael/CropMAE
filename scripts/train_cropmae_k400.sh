#!/usr/bin/env bash

#SBATCH --time="2-00:00:00"
#SBATCH --job-name=cropmae_k400
#SBATCH --output=./logs/%j_%x.out
#SBATCH --partition=gpu
#SBATCH --nodes=4   # specify number of nodes
#SBATCH --gpus-per-node=4 # specify number of GPUs per node
#SBATCH --cpus-per-gpu=8  # specify number of CPU cores per GPU
#SBATCH --mem-per-gpu=50G

echo "----------------- Environment ------------------"
module purge
module load EasyBuild/2023a
module load CUDA/12.2.0
module list

cd ~/CropMAE

date

# Delete /dev/shm/train if it exists
if [ -d /dev/shm/train ]; then
    rm -rf /dev/shm/train
fi

echo "Training CropMAE on K400..."

# Get the IP address and set port for MASTER node
echo "NODELIST="${SLURM_NODELIST}
master_ip=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "head node is ${master_ip}:${master_port}"


# Print the number of GPUs allocated per node (Optional)
if [ -z "$SLURM_GPUS_PER_NODE" ]; then
  GPUS_PER_NODE=1
else
  GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
fi

NUM_NODES=$SLURM_JOB_NUM_NODES
export GPUS=$SLURM_JOB_NUM_GPUS
export PORT=$master_port
export MASTER_ADDR=$master_ip

export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU

echo "Number of GPUs per node: $GPUS_PER_NODE"
echo "Number of nodes allocated: $NUM_NODES"

# Run the training script
# srun command and --nnodes, --nproc_per_node, --rdzv_backend, --rdzv_endpoint arguments are required for multi-node training
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$master_ip:$master_port \
    --rdzv-conf="join_timeout=3600" \
train_cropmae_k400.py \
--accum_iter=1 \
--batch_size=128 \
--betas=0.9,0.95 \
--blr=1e-4 \
--data_path=datasets/kinetics400/k400/k400_valid_files.txt \
--decoder_depth=4 \
--decoder_embed_dim=256 \
--decoder_num_heads=8 \
--device=cuda \
--epochs=400 \
--input_size=224 \
--log_dir=CropMAE/output_dir \
--log_per_epoch=2 \
--masking_ratio=0.985 \
--max_files=999999999999999 \
--min_lr_prop=0.0 \
--num_workers=$OMP_NUM_THREADS \
--output_dir=CropMAE/output_dir \
--patch_size=16 \
--random_area_min_global=0.50 \
--random_area_max_global=1.0 \
--random_area_min_local=0.3 \
--random_area_max_local=0.6 \
--random_aspect_ratio_max_global=1.3333333333333333 \
--random_aspect_ratio_max_local=1.3333333333333333 \
--random_aspect_ratio_min_global=0.75 \
--random_aspect_ratio_min_local=0.75 \
--repeated_sampling_factor=1 \
--save_every_n=25 \
--seed=42 \
--wandb=online \
--warmup_epochs_prop=0.05 \
--weight_decay=0.05 \
--wandb_run_name CropMAE_K400 \
--norm_pix_loss \
--crop_strategy GlobalToLocal \
# --slurm_davis

echo "Done training at $(date)"