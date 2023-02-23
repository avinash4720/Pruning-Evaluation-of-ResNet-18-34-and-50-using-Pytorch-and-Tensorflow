#!/bin/bash
#SBATCH --job-name=hpml
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=pytorch_resnet34.out
#SBATCH --mem=32GB
##SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
##SBATCH --account=ece-gy-9143-2022fa


module purge
module load anaconda3/2020.07
#module load python/intel/3.8.6
eval "$(conda shell.bash hook)"
conda activate assignment
/scratch/pp2603/envs_dirs/assignment/bin/python main.py --workers=4 --batchsize=128 --cuda --optimizer="SGD"
# /scratch/pp2603/envs_dirs/assignment/bin/python prune.py 
#/scratch/pp2603/hpml/ass5/q2.py --workers=4 --batchsize=256 --cuda --optimizer="SGD"
