#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --job-name=contractor
#SBATCH --mail-user=dalfaver@msu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

exec_path=~/decoder/examples

module load Python/3.11.3
source ~/.venv/decoding/bin/activate
python ${exec_path}/twod_contractor_test.py
