#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --output=%u_job_%j.out
#SBATCH --job-name=screate_in1k256

module load CUDA
module load Miniconda3
source ${EBROOTMINICONDA3}/bin/activate

conda activate reg

#256
python -u preprocessing/dataset_tools.py encode \
    --source=/mnt/beegfs/home/ac141281/edm2_c/datasets/img256.zip \
    --dest=/mnt/beegfs/home/ac141281/edm2_c/datasets/vae-sd-256

