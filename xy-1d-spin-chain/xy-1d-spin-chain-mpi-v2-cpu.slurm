#!/bin/sh

#SBATCH --job-name=netket-xy-1d-spin-chain
#SBATCH --partition=normal
## Separate output and error messages into 2 files.
#SBATCH --output=/scratch/%u/%x-%N-%j.out  # Output file
#SBATCH --error=/scratch/%u/%x-%N-%j.err   # Error file

## Disable email notifications
#SBATCH --mail-type=NONE
#SBATCH --mail-user=tcosgrov@gmu.edu

## Time the job needs. (default: see partition above)
#SBATCH --time=0-01:00  # Total time needed for job: Days-Hours:Minutes

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --array=1-5 # Array job with 5 tasks

set echo
umask 0027

## Load the relevant modules needed for the job
module load gnu/12.3.0
module load python/3.12.1-33

## Source the venv
source /home/tcosgrov/code/mb-psi-net/.venv/bin/activate

# Tell NetKet to NOT use sharding mode.
# Otherwise you get this: 
# File "/home/tcosgrov/code/mb-psi-net/.venv/lib/python3.12/site-packages/netket/operator/_discrete_operator.py", line 69, in get_conn_padded
# RuntimeError: When using Sharding mode, only jax operators are supported.
export NETKET_EXPERIMENTAL_SHARDING=0

srun -uv python /home/tcosgrov/code/mb-psi-net/xy-1d-spin-chain/xy-1d-spin-chain-mpi-v2.py
