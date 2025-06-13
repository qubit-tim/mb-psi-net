#!/bin/sh

## Give your job a name to distinguish it from other jobs you run.
#SBATCH --job-name=netket-ising-tutorial
#SBATCH --partition=normal        
## Separate output and error messages into 2 files.
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID
## for a more detailed filename --output=/scratch/%u/%x-%N-%j.out, --error=/scratch/%u/%x-%N-%j.err
#SBATCH --output=/scratch/%u/%x.out  # Output file
#SBATCH --error=/scratch/%u/%x.err   # Error file

## Slurm can send you updates via email
#SBATCH --mail-type=NONE         # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..
#SBATCH --mail-user=tcosgrov@gmu.edu     # Put your GMU email address here

## Specify how much memory your job needs. 
#SBATCH --mem=8G        # Total memory needed per task (units: K,M,G,T)

## Specify how much time your job needs. (default: see partition above)
#SBATCH --time=0-01:00  # Total time needed for job: Days-Hours:Minutes

set echo
umask 0027

## Load the relevant modules needed for the job
module load gnu/12.3.0
module load python/3.12.1-33

## Source the venv
source /home/tcosgrov/code/mb-psi-net/.venv/bin/activate

## Run your program or script
python /home/tcosgrov/code/mb-psi-net/netket-tutorials/1-ground-state-ising-model/ising-module.py
