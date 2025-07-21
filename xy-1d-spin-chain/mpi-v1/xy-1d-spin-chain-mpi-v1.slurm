#!/bin/sh

#SBATCH --job-name=netket-xy-1d-spin-chain
#SBATCH --partition=normal
## Separate output and error messages into 2 files.
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID
## for a more detailed filename --output=/scratch/%u/%x-%N-%j.out, --error=/scratch/%u/%x-%N-%j.err
#SBATCH --output=/scratch/%u/%x.out  # Output file
#SBATCH --error=/scratch/%u/%x.err   # Error file

## Slurm can send you updates via email
#SBATCH --mail-type=NONE
#SBATCH --mail-user=tcosgrov@gmu.edu

#SBATCH --hint=nomultithread  # Disable Hyperthreading

#SBATCH --mem=8G        # Total memory needed per task (units: K,M,G,T)

## Specify how much time your job needs. (default: see partition above)
#SBATCH --time=0-01:00  # Total time needed for job: Days-Hours:Minutes

#SBATCH --ntasks=4
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:4          # here you should insert the total number of gpus per node
#SBATCH --time=01:00:00

set echo
umask 0027

## Load the relevant modules needed for the job
module load gnu/12.3.0
module load python/3.12.1-33

## Source the venv
source /home/tcosgrov/code/mb-psi-net/.venv/bin/activate

# Tell NetKet to use sharding mode.
export NETKET_EXPERIMENTAL_SHARDING=1
# Tell Jax that we want to use GPUs. This is generally not needed but can't hurt
export JAX_PLATFORM_NAME=gpu

python /home/tcosgrov/code/mb-psi-net/xy-1d-spin-chain/xy-1d-spin-chain-mpi-v1.py