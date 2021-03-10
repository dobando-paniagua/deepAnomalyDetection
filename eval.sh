#!/usr/bin/env bash
#SBATCH --job-name=mscred_eval
#SBATCH --output=test%j.log
#SBATCH --error=test%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=obandod+Cluster@uni-hildesheim.de

# ## FOR GPU USE:
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
export HDF5_USE_FILE_LOCKING='FALSE'
source activate TEST

## Run the script
srun python evaluate.py
