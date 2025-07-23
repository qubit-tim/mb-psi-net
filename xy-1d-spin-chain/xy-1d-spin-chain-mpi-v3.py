import jax
import os

outdir = 'out/'
if os.getenv('SLURM_JOB_ID') is not None:
    # This is a SLURM job, initialize distributed JAX
    jax.distributed.initialize()
    outdir = '/scratch/tcosgrov/out/'  # Change this to your desired output directory on SLURM
    # Print this to verify correct setup
    print(f"[{jax.process_index()}/{jax.process_count()}] devices:", jax.devices(), flush=True)
    print(f"[{jax.process_index()}/{jax.process_count()}] local devices:", jax.local_devices(), flush=True)

import argparse
import json
import time
import warnings
import random

import jax.numpy as jnp
import matplotlib.pyplot as plt
import netket as nk

from dataclasses import dataclass, field
from flax import nnx
from netket.operator.spin import sigmax, sigmay, sigmaz

# Need to set the environment variable NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION=1 for netket to use the FFT-based autocorrelation
#os.environ['NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION'] = '1'

nk.config.netket_experimental_fft_autocorrelation = True  # Enable FFT-based autocorrelation
print("Using FFT-based autocorrelation:", nk.config.netket_experimental_fft_autocorrelation)

# The following classes are based on:
#  https://netket.readthedocs.io/en/latest/tutorials/gs-heisenberg.html

class Jastrow(nnx.Module):
    def __init__(self, N: int, *, rngs: nnx.Rngs):
        k1, k2 = jax.random.split(rngs.params())
        self.J = nnx.Param(0.01 * jax.random.normal(k1, (N, N),
                                                    dtype=jnp.complex128))

        self.v_bias = nnx.Param(0.01 * jax.random.normal(k2, (N, 1),
                                                         dtype=jnp.complex128))

    def __call__(self, x):
        x = x.astype(jnp.complex128)              # keep the dtypes aligned
        quad = jnp.einsum('...i,ij,...j->...', x, self.J.value, x)
        lin  = jnp.squeeze(x @ self.v_bias, -1)   # (...,N) @ (N,1) → (...,1)
        return quad + lin

class FFModel(nnx.Module):
    def __init__(self, N: int, *, rngs: nnx.Rngs):
        k1, k2 = jax.random.split(rngs.params())
        self.J = nnx.Param(0.01 * jax.random.normal(k1, (N, N),
                                                    dtype=jnp.complex128))

        self.v_bias = nnx.Param(0.01 * jax.random.normal(k2, (N, 1),
                                                         dtype=jnp.complex128))
        self.linear = nnx.Linear(
            in_features=N, 
            out_features=2 * N, 
            dtype=jnp.complex128, 
            param_dtype=jnp.complex128,
            rngs=rngs)

    def __call__(self, x: jax.Array):
        #x = x.astype(jnp.complex128)              # keep the dtypes aligned
        x = self.linear(x)
        x = nk.nn.activation.log_cosh(x)
        x = jnp.sum(x, axis=-1)
        return x
    
class FFModel2(nnx.Module):
    def __init__(self, N: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(
            in_features=N, 
            out_features=2 * N, 
            dtype=jnp.complex128, 
            param_dtype=jnp.complex128,
            rngs=rngs)
        self.linear2 = nnx.Linear(
            in_features=2 * N, 
            out_features=N, 
            dtype=jnp.complex128, 
            param_dtype=jnp.complex128,
            rngs=rngs)

    def __call__(self, x: jax.Array):
        x = self.linear1(x)
        x = nk.nn.activation.log_cosh(x)
        x = self.linear2(x)
        x = nk.nn.activation.log_cosh(x)
        x = jnp.sum(x, axis=-1)
        return x

# TODO: Convert these paramaters to optax parameters
@dataclass
class OptimizerConfig:
    learning_rate: float = 0.01  # Learning rate for optimizers
    momentum: float = 0.9  # Used for Momentum optimizer
    nesterov: bool = False  # Used for Momentum optimizer
    epscut: float = 1e-8  # Used for AdaGrad, RmsProp optimizers
    initial_accumulator_value: float = 0.1  # Used for AdaGrad optimizer
    b1: float = 0.9  # Used for Adam optimizer
    b2: float = 0.999  # Used for Adam optimizer
    eps: float = 1e-8  # Used for Adam optimizer
    beta: float = 0.9  # Used for RmsProp optimizer
    centered: bool = False  # Used for RmsProp optimizer
    

@dataclass
class RunConfig:
    model_type: str = "Jastrow"
    sampler_type: str = "MetropolisExchange"
    optimizer_type: str = "Sgd"
    preconditioner_type: str = "SR"
    variational_state_type: str = "MCState"
    driver_type: str = "VMC"
    hilbert: nk.hilbert.AbstractHilbert = None
    graph: nk.graph.Graph = None
    hamiltonian: nk.operator.LocalOperator = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    diag_shift: float = 0.1  # Used for SR preconditioner
    # Parameters for the model
    exact_sol: float = 0.0  # Exact solution energy
    output_dir: str = 'out/'
    n_samples: int = 1000
    n_iterations: int = 1000
    seed: int = 0
    alpha: float = 1.0  # Used for RBM and RBMSymm
    d_max: int = 1
    learning_rate: float = 0.01
    holomorphic: bool = False
    
    def getOutFileBase(self):
        out = str(self.output_dir) + f"{self.model_type}-n-{self.hilbert.size}-it-{self.n_iterations}-sa-{self.n_samples}-sd-{self.seed}"
        if self.model_type == "RBM" or self.model_type == "RBMSymm":
            out += f"-al-{self.alpha}"
        else:
            out += f"-lr-{self.learning_rate}"
        return out
    
    def __repr__(self):
        return (f"RunConfig(model_type={self.model_type}, sampler_type={self.sampler_type}, "
                f"optimizer_type={self.optimizer_type}, preconditioner_type={self.preconditioner_type}, "
                f"variational_state_type={self.variational_state_type}, driver_type={self.driver_type}, "
                f"hilbert={self.hilbert}, graph={self.graph}, hamiltonian={self.hamiltonian}, "
                f"exact_sol={self.exact_sol}, output_dir={self.output_dir}, n_samples={self.n_samples}, "
                f"n_iterations={self.n_iterations}, seed={self.seed}, d_max={self.d_max}, "
                f"learning_rate={self.learning_rate}, holomorphic={self.holomorphic})")

@dataclass
class GroundStateEnergy:
    mean: float = None
    estimate_error_of_mean: float = None
    variance: float = None
    tau_correlation: float = None  # τ correlation time
    r_hat: float = None  # R̂ statistic for convergence

@dataclass
class ModelResults:
    ground_state_energy: GroundStateEnergy = field(default_factory=GroundStateEnergy)
    num_parameters: int = 0
    seconds_to_calculate: float = 0.0
    output_directory: str = ''
    model_type: str = ''
    hilbert_size: int = 0
    num_iterations: int = 0
    num_samples: int = 0
    seed: int = 0
    learning_rate: float = 0.0
    alpha: float = 0.0
    exact_solution_energy: float = 0.0
    preconditioner_type: str = ''
    sampler_type: str = ''
    optimizer_type: str = ''
    variational_state_type: str = ''
    graph_type: str = ''
    graph_nodes: int = 0
    graph_edges: int = 0
    
    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__,
            sort_keys=True,
            indent=4)
    
    def __repr__(self):
        return (f"ModelResults(ground_state_energy={self.ground_state_energy}, num_parameters={self.num_parameters}, "
                f"seconds_to_calculate={self.seconds_to_calculate}, output_directory={self.output_directory}, "
                f"model_type={self.model_type}, hilbert_size={self.hilbert_size}, num_iterations={self.num_iterations}, "
                f"num_samples={self.num_samples}, seed={self.seed}, learning_rate={self.learning_rate}, "
                f"alpha={self.alpha}, exact_solution_energy={self.exact_solution_energy}, "
                f"preconditioner_type={self.preconditioner_type}, sampler_type={self.sampler_type}, "
                f"optimizer_type={self.optimizer_type}, variational_state_type={self.variational_state_type}, "
                f"graph_type={self.graph_type}, graph_nodes={self.graph_nodes}, graph_edges={self.graph_edges})")


class ModelRun():
    def __init__(self, config: RunConfig):
        self.config = config
        self.results = ModelResults()
        self.model = self._setup_model()
        self.sampler = self._setup_sampler()
        self.optimizer = self._setup_optimizer()
        self.preconditioner = self._setup_preconditioner()
        self.variational_state = self._setup_variational_state()
        self.driver = self._setup_driver()

    def _setup_model(self):
        match self.config.model_type:
            case "Jastrow":
                return Jastrow(N=self.config.hilbert.size, rngs=nnx.Rngs(self.config.seed))
            case "RBM":
                return nk.models.RBM(alpha=self.config.alpha, param_dtype=jnp.complex128)
            case "RBMSymm":
                return nk.models.RBMSymm(symmetries=self.config.graph.translation_group(), alpha=self.config.alpha, param_dtype=jnp.complex128)
            case "FF":
                return FFModel(N=self.config.hilbert.size, rngs=nnx.Rngs(self.config.seed))
            case "FF2":
                return FFModel2(N=self.config.hilbert.size, rngs=nnx.Rngs(self.config.seed))
            case _:
                raise ValueError(f"Model type '{self.config.model_type}' is not recognized.")
    
    def _setup_sampler(self):
        match self.config.sampler_type:
            case "ExactSampler":
                raise NotImplementedError("ExactSampler is not yet implemented.")
            case "MetropolisLocal":
                return nk.sampler.MetropolisLocal(hilbert=self.config.hilbert)
            case "MetropolisExchange":
                return nk.sampler.MetropolisExchange(hilbert=self.config.hilbert, graph=self.config.graph, d_max=self.config.d_max)
            case "MetropolisHamiltonian":
                return nk.sampler.MetropolisHamiltonian(hilbert=self.config.hilbert, hamiltonian=self.config.hamiltonian)
            case "MetropolisGaussian":
                raise NotImplementedError("MetropolisGaussian is not yet implemented.")
            case "MetropolisAdjustedLangevin":
                raise NotImplementedError("MetropolisAdjustedLangevin is not yet implemented.")
            case "MetropolisFermionHop":
                raise NotImplementedError("MetropolisFermionHop is not yet implemented.")
            case _:
                raise ValueError(f"Sampler type '{self.config.sampler_type}' is not recognized.")
    
    def _setup_optimizer(self):
        # TODO: Convert these to optax optimizers
        match self.config.optimizer_type:
            case "Sgd":
                return nk.optimizer.Sgd(learning_rate=self.config.learning_rate)
            case "Momentum":
                return nk.optimizer.Momentum(learning_rate=self.config.learning_rate, beta=self.config.optimizer.momentum)
            case "Adam":
                return nk.optimizer.Adam(learning_rate=self.config.learning_rate, b1=self.config.optimizer.b1, b2=self.config.optimizer.b2, eps=self.config.optimizer.eps)
            case "AdaGrad":
                return nk.optimizer.AdaGrad(learning_rate=self.config.learning_rate, epscut=self.config.optimizer.epscut, initial_accumulator_value=self.config.optimizer.initial_accumulator_value)
            case "RmsProp":
                return nk.optimizer.RmsProp(learning_rate=self.config.learning_rate, beta=self.config.optimizer.beta, epscut=self.config.optimizer.epscut, centered=self.config.optimizer.centered)
            case _:
                raise ValueError(f"Optimizer type '{self.config.optimizer_type}' is not recognized.")
    
    def _setup_preconditioner(self):
        match self.config.preconditioner_type:
            case "SR":
                return nk.optimizer.SR(diag_shift=self.config.diag_shift, holomorphic=self.config.holomorphic)
            case _:
                raise ValueError(f"Preconditioner type '{self.config.preconditioner_type}' is not recognized.")
    
    def _setup_variational_state(self):
        if self.sampler is None or self.model is None:
            raise ValueError("Sampler and model must be set before setting up the variational state.")
        match self.config.variational_state_type:
            case "MCState":
                return nk.vqs.MCState(self.sampler, self.model, n_samples=self.config.n_samples)
            case "MCMixedState":
                raise NotImplementedError("MCMixedState is not yet implemented.")
            case "FullSumState":
                raise NotImplementedError("FullSumState is not yet implemented.")
            case _:
                raise ValueError(f"Variational state type '{self.config.variational_state_type}' is not recognized.")
    
    def _setup_driver(self):
        if self.config.hamiltonian is None or self.optimizer is None or self.preconditioner is None or self.variational_state is None:
            raise ValueError("Hamiltonian, optimizer, preconditioner, and variational state must be set before setting up the driver.")
        match self.config.driver_type:
            case "VMC":
                return nk.VMC(
                    hamiltonian=self.config.hamiltonian,
                    optimizer=self.optimizer,
                    preconditioner=self.preconditioner,
                    variational_state=self.variational_state)
            case "SteadyState":
                raise NotImplementedError("SteadyState is not yet implemented.")
            case "QSR":
                raise NotImplementedError("QSR is not yet implemented.")
            case _:
                raise ValueError(f"Driver type '{self.config.driver_type}' is not recognized.")

    def run(self, n_iterations: int = 0):
        n_iterations = n_iterations if n_iterations > 0 else self.config.n_iterations
        if n_iterations <= 0:
            raise ValueError("Number of iterations must be greater than 0.")
            
        if self.driver is None:
            raise ValueError("Driver must be set before running the model.")
        
        if self.config.model_type == "RBM" or self.config.model_type == "RBMSymm":
            print(f'Running {self.config.model_type} Model: {self.config.hilbert.size} nodes, {n_iterations} iterations, {self.config.n_samples} samples, seed {self.config.seed}, alpha {self.config.alpha}')
        else:
            print(f'Running {self.config.model_type} Model: {self.config.hilbert.size} nodes, {n_iterations} iterations, {self.config.n_samples} samples, seed {self.config.seed}, learning rate {self.config.learning_rate}')

        start = time.time()
        out = self.config.getOutFileBase()
        
        print(f'Output base name: {out}')
        
        self.driver.run(out=out, n_iter=n_iterations)
        end = time.time()

        print(f'{self.config.model_type} Model calculation finished')
        print(f'{self.config.model_type} Model Results:')
        print('- Ground state energy approximation: ', self.driver.energy)
        if self.config.exact_sol != 0:
            error = jnp.abs((self.driver.energy.mean - self.config.exact_sol) / self.config.exact_sol)
            print('- Relative Error: ', error)
            print('- Percentage Error: ', error * 100, '%')
        print('- Number of Parameters: ', nk.jax.tree_size(self.variational_state.parameters))
        print('- Seconds to perform the calculation:', end - start)
        print()
        # Store results
        self.results.ground_state_energy.mean = float(self.driver.energy.mean.real)
        self.results.ground_state_energy.estimate_error_of_mean = float(self.driver.energy.error_of_mean.real)
        self.results.ground_state_energy.variance = float(self.driver.energy.variance.real)
        self.results.ground_state_energy.tau_correlation = float(self.driver.energy.tau_corr.real)
        self.results.ground_state_energy.r_hat = float(self.driver.energy.R_hat.real)
        self.results.num_parameters = nk.jax.tree_size(self.variational_state.parameters)
        self.results.seconds_to_calculate = end - start
        self.results.output_directory = out
        self.results.model_type = self.config.model_type
        self.results.hilbert_size = self.config.hilbert.size
        self.results.num_iterations = n_iterations
        self.results.num_samples = self.config.n_samples
        self.results.seed = self.config.seed
        self.results.learning_rate = self.config.learning_rate
        self.results.alpha = self.config.alpha
        self.results.exact_solution_energy = self.config.exact_sol
        self.results.preconditioner_type = self.config.preconditioner_type
        self.results.sampler_type = self.config.sampler_type
        self.results.optimizer_type = self.config.optimizer_type
        self.results.variational_state_type = self.config.variational_state_type
        self.results.graph_type = type(self.config.graph).__name__
        self.results.graph_nodes = self.config.graph.n_nodes
        self.results.graph_edges = self.config.graph.n_edges
        print(self.results.toJSON())
        print()
        self.write_results()
    
    def write_results(self, filename: str = None):
        if filename is None:
            filename = self.config.getOutFileBase() + '-results.json'
        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines(self.results.toJSON())
        print(f'Results written to {filename}')

def main(args):
    print(args)
    # These won't change during the run
    model = args.model  # Model type to use
    pbc = args.pbc # Periodic Boundary Conditions
    seed = args.seed  # Random seed for reproducibility
    if seed < 0:
        seed = random.randint(0, 2**32)
    N = args.nodes  # Number of nodes in the spin chain
    exact_gs_energy = -1.2732395447351628 * N  # Exact ground state energy for the 1D XY spin chain
    n_s = args.samples  # Number of samples to use
    n_i = args.iterations  # Number of iterations to run
    l_r = args.learning_rate  # Learning rate for the optimizer
    alpha = args.alpha  # Alpha parameter for the RBM model
    d_m = args.d_max # Maximum distance for the Metropolis sampler
    
    # We are running locally, so let's use some default values
    if os.getenv('SLURM_JOB_ID') is None:
        model = model if model != None else "RBM" # Change this to the model you want to test locally
        pbc = True
        N = 10
        exact_gs_energy = -1.2732395447351628 * N
        n_s = 1000
        n_i = 600
        l_r = 0.001
        alpha = 1.0

    if args.ignore_warnings:
        warnings.filterwarnings("ignore", category=UserWarning, module="netket")

    print('### 1D XY Spin Chain Calculations')
    print('- Random seed for rngs:', seed)
    print('- Periodic Boundary Conditions:', pbc)
    print('- The analytical ground state * 4: U/N = -4/pi :', -4/jnp.pi)
    print('- The analytical ground state (U/N): -1/pi = ', -1/jnp.pi)
    print()
    
    print(f'## Run Parameters: N={N}, Exact GS Energy={exact_gs_energy}, Samples={n_s}, Iterations={n_i}, Learning Rate={l_r}, Alpha={alpha}')
    g = nk.graph.Hypercube(length=N, n_dim=1, pbc=pbc)
    hi = nk.hilbert.Spin(s=0.5, total_sz=0, N=g.n_nodes)
    ha = sum([(sigmax(hi, i) * sigmax(hi, i+1 if i+1 < N else 0) + sigmay(hi, i) * sigmay(hi, i+1 if i+1 < N else 0)) for i in range(0,N)])
    if not pbc:
        ha = sum([(sigmax(hi, i) * sigmax(hi, i+1) + sigmay(hi, i) * sigmay(hi, i+1)) for i in range(0,N-1)])
                        
    run_config = RunConfig(
        model_type=model,
        sampler_type="MetropolisExchange",
        optimizer_type="Sgd",
        preconditioner_type="SR",
        variational_state_type="MCState",
        driver_type="VMC",
        hilbert=hi,
        graph=g,
        hamiltonian=ha,
        exact_sol=exact_gs_energy,
        output_dir=outdir,
        n_samples=n_s,
        n_iterations=n_i,
        seed=seed,
        alpha=alpha,
        d_max=d_m,
        learning_rate=l_r
    )

    ModelRun(config=run_config).run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XY 1D Spin Chain Script.")

    # Positional arguments - these are required
    #parser.add_argument("model", type=str, help="The name of the model to use.")

    # Optional arguments
    parser.add_argument("--model", type=str, help="The name of the model to use.")
    parser.add_argument("--nodes", type=int, default=10, help="Number of nodes in the spin chain.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to use.")
    parser.add_argument("--iterations", type=int, default=600, help="Number of iterations to run.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha parameter for the RBM model.")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed for reproducibility.")
    parser.add_argument("--d_max", type=int, default=1, help="Maximum distance for the Metropolis sampler.")
    parser.add_argument("--pbc", type=bool, default=True, help="Use periodic boundary conditions.")
    parser.add_argument("--ignore_warnings", type=bool, default=True, help="Ignore specific warnings.")

    main(parser.parse_args())
