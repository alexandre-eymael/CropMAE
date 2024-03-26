#!/usr/bin/env bash

#SBATCH --time="0-01:00:00"
#SBATCH --job-name=seg
#SBATCH --partition=gpu
#SBATCH --nodes=1   # specify number of nodes
#SBATCH --gpus-per-node=1 # specify number of GPUs per node
#SBATCH --cpus-per-gpu=8  # specify number of CPU cores per GPU
#SBATCH --mem-per-gpu=80G

echo "----------------- Environment ------------------"
module purge
module load EasyBuild/2023a
module load CUDA/12.2.0
module list

micromamba activate CropMAE

cd ~/CropMAE

date

python3 -m downstreams.propagation.start \
    results \
    399 \
    $1