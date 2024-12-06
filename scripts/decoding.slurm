#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=deocder
#SBATCH --mail-user=dalfaver@msu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mem=36G

module load Python/3.11.3
source /mnt/home/dalfaver/.venv/decoding/bin/activate

exec_dir=/mnt/home/dalfaver/decoder/examples
python ${exec_dir}/decode_errors.py 10000 10 logical_error_rates.csv
