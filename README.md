# mb-psi-net
Many Body 𝛹 Netket Project

# OCR Instructions
1. module load gnu/12.3.0
1. module load python/3.12.1-33
1. source .venv/bin/activate

# Freezing requirements
pip freeze | sed 's/==/>=/g' > requirements.txt

# Pip requirements update
pip install -r requirements.txt --upgrade
