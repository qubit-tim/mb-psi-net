# mb-psi-net
Many Body 𝛹 Netket Project

# GMU OCR Hopper Instructions
1. module load gnu/12.3.0
1. module load python/3.12.1-33
1. source .venv/bin/activate

# Freezing requirements
pip freeze | sed 's/==/>=/g' > requirements.txt

# Pip requirements update
pip install -r requirements.txt --upgrade

# SLURM Commands
1. sgpu - shows gpuq stats (Total, Allocated, Idle)
    1. -v adds detailed stats
1. squeue -u $USER - 
1. sreport cluster utilization -T 'ALL'
1. sinfo -N -o "%N %c" | grep hop | sort -u | awk -F' ' '{sum += $2} END {print sum}'
    1. This will show available CPU cores (I think)

TODO: Find a command to view available CPU nodes / cores

# Job Checkpoints
1. DMTCP - https://dmtcp.sourceforge.io/index.html
1. https://wiki.orc.gmu.edu/mkdocs/Creating_Checkpoints_%28DMTCP%29/

