# https://netket.readthedocs.io/en/latest/docs/clusters.html

#SBATCH --job-name=test
#SBATCH --output=test_%j.txt
#SBATCH --hint=nomultithread  # Disable Hyperthreading

#SBATCH --ntasks=4
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:4          # here you should insert the total number of gpus per node
#SBATCH --time=01:00:00

module purge

# Load the same packages you used during installation. In our case that is
module load gcc/12.2.0 anaconda-py3 

# Load the conda environment or equivalent
conda activate ENV_NAME

# Tell NetKet to use sharding mode.
export NETKET_EXPERIMENTAL_SHARDING=1
# Tell Jax that we want to use GPUs. This is generally not needed but can't hurt
export JAX_PLATFORM_NAME=gpu

srun python yourscript.py