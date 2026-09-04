#!/bin/bash
#SBATCH --job-name=grid_retrieval
#SBATCH --account=def-rdoyon
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=0-20:00

#SBATCH --output=/home/mathisb/scratch/HRS_retrievals/grid_retrievals/slurm_outputs/grid_retrieval_%j.out

#SBATCH --mail-user=mathis.bouffard@umontreal.ca
#SBATCH --mail-type=BEGIN,END,FAIL

source /home/mathisb/HighRes/bin/activate

# Make sure it knows where to look for starships
export PYTHONPATH=/home/mathisb/Github/starships:$PYTHONPATH

cd /home/mathisb/Github/HRS_analysis/retrievals
python grid_retrieval.py