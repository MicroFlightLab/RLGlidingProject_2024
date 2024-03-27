#!/bin/bash
#SBATCH --job-name=gliding
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time=50:00:00
#SBATCH --mail-user=yoav.flato@mail.huji.ac.il
#SBATCH --mail-type=END
hyper_choice=$1


echo "hyperparameter choice $hyper_choice"
source /cs/labs/tsevi/yoavflato/gliding_venv/bin/activate.csh
module load torch
python3 /cs/labs/tsevi/yoavflato/GlidingProject/seed_main.py ${SLURM_ARRAY_TASK_ID} --hyperparameter=$hyper_choice