#!/bin/bash
#SBATCH --partition=normal                 # will run on any cpus in the 'normal' partition
#SBATCH --job-name=python-cpu
## Separate output and error messages into 2 files.
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID
#SBATCH --output=/scratch/%u/%x-%N-%j.out  # Output file
#SBATCH --error=/scratch/%u/%x-%N-%j.err   # Error file
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1                   # up to 48 per node
#SBATCH --mem-per-cpu=3500M                 # memory per CORE; maximum is 180GB per node
#SBATCH --export=ALL
#SBATCH --time=0-01:00:00                   # set to 1hr; please choose carefully

set echo
umask 0027

module load gnu10
module load python                          # load the recommended Python version

python myscript.py                          # execute your Python script