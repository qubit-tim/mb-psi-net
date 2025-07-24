import random

from jinja2 import Environment, FileSystemLoader


env = Environment(loader=FileSystemLoader('run-templates/'))

template = env.get_template('cpu-run.jinja2')

if __name__ == "__main__":
    # Example context for rendering the template
    seed = random.randint(0, 2**32)  # Random seed for reproducibility
    
    # These will be iterated over
    models = ['Jastrow', 'RBM', 'FF', 'FF2']
    node_counts = [10, 20, 40, 80, 160]
    samples = [1000, 2000, 4000, 8000, 16000]
    iterations = [300, 600, 1200, 2400, 4800, 9600]
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    model = 'RBM'
    nodes = 10
    samples = 1000
    iterations = 600
    learning_rate = 0.001
    alpha = 1.0
    seed = 42
    args = f'--model {model} --nodes {nodes} --samples {samples} --iterations {iterations} --learning-rate {learning_rate} --alpha {alpha} --seed {seed}'
    context = {
        'job_name': 'netket-xy-1d-spin-chain',
        'partition': 'normal',
        'output_file': '/scratch/%u/%x-%N-%j.out',
        'error_file': '/scratch/%u/%x-%N-%j.err',
        'mail_type': 'NONE',
        'mail_user': 'tcosgrov@gmu.edu',
        'time': '0-01:00',
        #'nodes': 1,
        'cpus_per_task': 1,
        'mem_per_cpu': '4G',
        #'array': '1-10',
        'modules': ['gnu/12.3.0', 'python/3.12.1-33'],
        'venv_source': '/home/tcosgrov/code/mb-psi-net/.venv/bin/activate',
        'exports': ['NETKET_EXPERIMENTAL_SHARDING=0'],
        'python_script': '/home/tcosgrov/code/mb-psi-net/xy-1d-spin-chain/xy-1d-spin-chain-mpi-v3.py',
        'python_args': args,
    }
    
    job_file = '~/cpu-run.slurm'
    with open(job_file, 'w') as fh:
        rendered_script = template.render(context)
        fh.write(rendered_script)
        print(rendered_script)
        print(f"Job script written to {job_file}")
    