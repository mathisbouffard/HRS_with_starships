#!/bin/bash
#SBATCH --job-name=grid_injection
#SBATCH --account=def-rdoyon
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=0-12:00

#SBATCH --output=/home/mathisb/scratch/HRS_models/grid_inj_slurm_outputs/grid_injection_%j.out

#SBATCH --mail-user=mathis.bouffard@umontreal.ca
#SBATCH --mail-type=BEGIN,END,FAIL

source /home/mathisb/HighRes/bin/activate

# Make sure it knows where to look for starships
export PYTHONPATH=/home/mathisb/Github/starships:$PYTHONPATH

cd /home/mathisb/Github/HRS_analysis/grid_injections
python grid_injection.py