#!/bin/bash
#SBATCH --job-name=gliding
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=01:00:00
#SBATCH --mail-user=yoav.flato@mail.huji.ac.il
#SBATCH --mail-type=END

source /sci/labs/tsevi/yoavflato/glider-venv/bin/activate.csh
pip install -r /sci/labs/tsevi/yoavflato/GlidingProject/requirements.txt