#!/bin/bash
#SBATCH --job-name=hpml
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=tensor_project.out
#SBATCH --mem=32GB
##SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
##SBATCH --account=ece-gy-9143-2022fa


module purge
module load anaconda3/2020.07
#module load python/intel/3.8.6
eval "$(conda shell.bash hook)"
conda activate assignment

/scratch/pp2603/envs_dirs/assignment/bin/python main.py 
