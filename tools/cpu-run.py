import itertools
import os
import platform
import random
import time

from dataclasses import dataclass, field
from jinja2 import Environment, FileSystemLoader

@dataclass
class JobConfig:
    job_name: str = None
    partition: str = 'normal'
    output_file: str ='/scratch/%u/%x-%N-%j.out'
    error_file: str = '/scratch/%u/%x-%N-%j.err'
    mail_type: str = 'NONE'
    mail_user: str = '$USER@gmu.edu'
    run_limit: str = None
    cpus_per_task: int = 1
    mem_per_cpu: str = '4G'
    modules: list = field(default_factory=lambda: ['gnu/12.3.0', 'python/3.12.1-33'])
    venv_source: str = None
    exports: list = field(default_factory=list)
    python_script: str = None
    python_args: str = None


env = Environment(loader=FileSystemLoader('run-templates/'))

template = env.get_template('cpu-run.jinja2')

def run_job(config: JobConfig):
    """
    Render the job script from the given configuration and submit it.
    """
    rendered_script = template.render(config.__dict__)
    if platform.system() != 'Linux':
        print("Running in a non-Linux environment. Not writing to file.")
        print(rendered_script)
        return
    
    job_file = '~/cpu-run.slurm'
    with open(os.path.expanduser(job_file), 'w') as fh:
        fh.write(rendered_script)
        print(rendered_script)
        print(f'Job script written to {job_file}')

    os.system(f'sbatch {job_file}')
    print(f'Job submitted with sbatch {job_file}')

def sleep():
    """
    Sleep for a random duration between 0.1 and 0.5 seconds.
    """
    sleep_duration = random.uniform(0.1, 0.5)
    print(f"Sleeping for {sleep_duration:.3f} seconds...")
    time.sleep(sleep_duration)
    print("Finished sleeping.")

def main():
    # Example context for rendering the template
    seed = random.randint(0, 2**32)  # Random seed for reproducibility
    
    # These will be iterated over
    alpha_models = ['RBM']
    learning_rate_models = ['Jastrow', 'FF', 'FF2']
    node_counts = [10, 20] #, 40, 80, 160]
    samples_list = [1000] #, 2000, 4000, 8000, 16000]
    iterations_list = [300] #, 600, 1200, 2400, 4800, 9600]
    #learning_rates = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    learning_rates = [0.001]
    #alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    alphas = [1.0]
    seed = 42
    cfg = JobConfig(
            job_name='netket-xy-1d-spin-chain',
            partition='normal',
            output_file='/scratch/%u/%x-%N-%j.out',
            error_file='/scratch/%u/%x-%N-%j.err',
            mail_type='NONE',
            mail_user='tcosgrov@gmu.edu',
            run_limit='0-01:00',
            cpus_per_task=1,
            mem_per_cpu='4G',
            modules=['gnu/12.3.0', 'python/3.12.1-33'],
            venv_source='/home/tcosgrov/code/mb-psi-net/.venv/bin/activate',
            exports=['NETKET_EXPERIMENTAL_SHARDING=0'],
            python_script='/home/tcosgrov/code/mb-psi-net/xy-1d-spin-chain/xy-1d-spin-chain-mpi-v3.py',
        )

    for combo in itertools.product(alpha_models, node_counts, samples_list, iterations_list, alphas):
        model, nodes, samples, iterations, alpha = combo
        args = f'--model {model} --nodes {nodes} --samples {samples} --iterations {iterations} --alpha {alpha} --seed {seed}'
        cfg.python_args = args
        run_job(cfg)

    for combo in itertools.product(learning_rate_models, node_counts, samples_list, iterations_list, learning_rates):
        model, nodes, samples, iterations, learning_rate = combo
        args = f'--model {model} --nodes {nodes} --samples {samples} --iterations {iterations} --learning-rate {learning_rate} --seed {seed}'
        cfg.python_args = args
        run_job(cfg)


if __name__ == "__main__":
    main()
