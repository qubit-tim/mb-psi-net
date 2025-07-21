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

import time
import warnings
import random

import jax.numpy as jnp
import matplotlib.pyplot as plt
import netket as nk

from dataclasses import dataclass, field
from flax import nnx
from netket.operator.spin import sigmax, sigmay, sigmaz

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
    exact_sol: float = 0.0
    output_dir: str = 'out/'
    n_samples: int = 1000
    n_iterations: int = 1000
    seed: int = 0
    alpha: float = 1.0  # Used for RBM and RBMSymm
    d_max: int = 1
    learning_rate: float = 0.01
    holomorphic: bool = False
    
    def __repr__(self):
        return (f"RunConfig(model_type={self.model_type}, sampler_type={self.sampler_type}, "
                f"optimizer_type={self.optimizer_type}, preconditioner_type={self.preconditioner_type}, "
                f"variational_state_type={self.variational_state_type}, driver_type={self.driver_type}, "
                f"hilbert={self.hilbert}, graph={self.graph}, hamiltonian={self.hamiltonian}, "
                f"exact_sol={self.exact_sol}, output_dir={self.output_dir}, n_samples={self.n_samples}, "
                f"n_iterations={self.n_iterations}, seed={self.seed}, d_max={self.d_max}, "
                f"learning_rate={self.learning_rate}, holomorphic={self.holomorphic})")

class ModelRun():
    def __init__(self, config: RunConfig):
        self.config = config
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
            print(f'Running {self.config.model_type} Model: {n_iterations} iterations, {self.config.n_samples} samples, seed {self.config.seed}, alpha {self.config.alpha}')
        else:
            print(f'Running {self.config.model_type} Model: {n_iterations} iterations, {self.config.n_samples} samples, seed {self.config.seed}')

        start = time.time()
        out = str(self.config.output_dir) + f"{self.config.model_type}-it-{n_iterations}-sa-{self.config.n_samples}-sd-{self.config.seed}"
        print(f'Output directory: {out}')
        if self.config.model_type == "RBM" or self.config.model_type == "RBMSymm":
            out += f"-al-{self.config.alpha}"
        else:
            out += f"-lr-{self.config.learning_rate}"
        
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

def main():
    # These won't change during the run
    pbc = True # Periodic Boundary Conditions
    ignore_warnings = True # Ignore specific warnings
    seed = random.randint(0, 2**32)  # Random seed for reproducibility
    
    # These will be iterated over
    node_counts = jnp.array([10, 20, 40, 80, 160])
    exact_gs_energy_multiple = -1.2732395447351628
    samples = jnp.array([1000, 2000, 4000, 8000, 16000])
    iterations = jnp.array([300, 600, 1200, 2400, 4800])
    learning_rates = jnp.array([0.001, 0.01, 0.1, 1, 10])
    alphas = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0])
    d_m = 1 # Maximum distance for the Metropolis sampler
    
    if ignore_warnings:
        warnings.filterwarnings("ignore", category=UserWarning, module="netket")

    if len(learning_rates) != len(alphas):
        raise ValueError("The number of learning rates must match the number of alphas.")

    print('### 1D XY Spin Chain Calculations')
    print('- Random seed for rngs:', seed)
    print('- Periodic Boundary Conditions:', pbc)
    print('- The analytical ground state * 4: U/N = -4/pi :', -4/jnp.pi)
    print('- The analytical ground state (U/N): -1/pi = ', -1/jnp.pi)
    print()
    for N in node_counts:
        for n_s in samples:
            for n_i in iterations:
                for i in range(len(learning_rates)): # pylint: disable=consider-using-enumerate
                    l_r = learning_rates[i]
                    alpha = alphas[i]
                    N = int(N)
                    exact_gs_energy = N * exact_gs_energy_multiple
                    n_s = int(n_s)
                    n_i = int(n_i)
                    l_r = float(l_r)
                    alpha = float(alpha)
                    print(f'## Run Parameters: N={N}, Exact GS Energy={exact_gs_energy}, Samples={n_s}, Iterations={n_i}, Learning Rate={l_r}, Alpha={alpha}')
                    g = nk.graph.Hypercube(length=N, n_dim=1, pbc=pbc)
                    hi = nk.hilbert.Spin(s=0.5, total_sz=0, N=g.n_nodes)
                    ha = sum([(sigmax(hi, i) * sigmax(hi, i+1 if i+1 < N else 0) + sigmay(hi, i) * sigmay(hi, i+1 if i+1 < N else 0)) for i in range(0,N)])
                    if not pbc:
                        ha = sum([(sigmax(hi, i) * sigmax(hi, i+1) + sigmay(hi, i) * sigmay(hi, i+1)) for i in range(0,N-1)])

                    rbm_config = RunConfig(
                        model_type="RBM",
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
                    rbm_model_run = ModelRun(config=rbm_config)
                    rbm_model_run.run()

                    jastrow_config = RunConfig(
                        model_type="Jastrow",
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
                        d_max=d_m,
                        learning_rate=l_r
                    )
                    jastrow_model_run = ModelRun(config=jastrow_config)
                    jastrow_model_run.run()

                    ffn_config = RunConfig(
                        model_type="FF",
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
                        d_max=d_m,
                        learning_rate=l_r
                    )
                    ffn_model_run = ModelRun(config=ffn_config)
                    ffn_model_run.run()

                    ffn2_config = RunConfig(
                        model_type="FF2",
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
                        d_max=d_m,
                        learning_rate=l_r
                    )
                    ffn2_model_run = ModelRun(config=ffn2_config)
                    ffn2_model_run.run()

if __name__ == "__main__":
    main()