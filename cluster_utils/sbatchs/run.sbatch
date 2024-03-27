#!/bin/bash
#SBATCH --job-name=gliding
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time=50:00:00
#SBATCH --mail-user=yoav.flato@mail.huji.ac.il
#SBATCH --mail-type=END
hyper_choice=$1
algorithm_choice=$2
description_choice=$3


echo "hyperparameter choice $hyper_choice"
echo "algorithm choice algorithm_choice"
echo "description choice description_choice"
source /cs/labs/tsevi/yoavflato/gliding_venv/bin/activate.csh
module load torch
python3 /cs/labs/tsevi/yoavflato/GlidingProject/main.py $algorithm_choice $description_choice --hyperparameter=$hyper_choice