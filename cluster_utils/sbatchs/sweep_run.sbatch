#!/bin/bash
#SBATCH --job-name=gliding
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:4
#SBATCH --time=50:00:00
#SBATCH --mail-user=yoav.flato@mail.huji.ac.il
#SBATCH --mail-type=END
hyper_choice=$1
num_sweep=$2


echo "hyperparameter choice $hyper_choice"
echo "number of runs in sweep $num_sweep"
source /cs/labs/tsevi/yoavflato/gliding_venv/bin/activate.csh
module load torch
python3 /cs/labs/tsevi/yoavflato/GlidingProject/main.py "sweep" "sweep_run" --count_sweep $((num_sweep)) --hyperparameter=$hyper_choice