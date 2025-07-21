import jax
import json
import os
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
        
        print(f'### Running {self.config.model_type} Model')
        
        start = time.time()
        out = str(self.config.output_dir) + f"{self.config.model_type}-it-{n_iterations}-sa-{self.config.n_samples}-sd-{self.config.seed}"
        if self.config.model_type == "RBM" or self.config.model_type == "RBMSymm":
            out += f"-al-{self.config.alpha}"
        self.driver.run(out=out, n_iter=n_iterations)
        end = time.time()

        print(f'### {self.config.model_type} Model calculation finished')
        print(f'### {self.config.model_type} Model Results')
        print('- Ground state energy approximation: ', self.driver.energy)
        if self.config.exact_sol != 0:
            error = jnp.abs((self.driver.energy.mean - self.config.exact_sol) / self.config.exact_sol)
            print('- Relative Error: ', error)
            print('- Percentage Error: ', error * 100, '%')
        print('- Number of Parameters: ', nk.jax.tree_size(self.variational_state.parameters))
        print('- Seconds to perform the calculation:', end - start)
        print()

# Initialization parameters
N = 10
pbc = True # Periodic Boundary Conditions
show_nk_graph = False # Show the NetKet graph
print_hamiltonian = False # Print the Hamiltonian to dense format; useful for troubleshooting
heisenberg_hamiltonian = False # Use the Heisenberg Hamiltonian instead of the XY Hamiltonian; useful for troubleshooting
exact_gs_energy = -1.2732395447351628 * N # Set this to the exact ground state energy if known to avoid redoing the calculations
ignore_warnings = True # Ignore specific warnings

# Run parameters
n_s = 1000
n_i = 300
# Ok, the Rngs is actually the random number generator seed for the parameters of the model.
# https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/rnglib.html
# This has been renammed as such
#seed = 1 
seed = random.randint(0, 2**32)  # Random seed for reproducibility
print('### Random seed for rngs:', seed)
d_m = 1
l_r = 0.01
alpha = 1.0  # RBM alpha

if ignore_warnings:
    warnings.filterwarnings("ignore", category=UserWarning, module="netket")

print('### 1D XY Spin Chain Calculation')
print('### N: ', N)

g = nk.graph.Hypercube(length=N, n_dim=1, pbc=pbc)
if show_nk_graph:
    print('### NetKet Graph')
    print(g._sites)
    print(g.adjacency_list())
    print(g.n_edges)
    print(g.n_nodes)
    g.draw()

hi = nk.hilbert.Spin(s=0.5, total_sz=0, N=g.n_nodes)

ha = sum([(sigmax(hi, i) * sigmax(hi, i+1 if i+1 < N else 0) + sigmay(hi, i) * sigmay(hi, i+1 if i+1 < N else 0)) for i in range(0,N)])
if not pbc:
    ha = sum([(sigmax(hi, i) * sigmax(hi, i+1) + sigmay(hi, i) * sigmay(hi, i+1)) for i in range(0,N-1)])

if heisenberg_hamiltonian:
    print('### Using the Heisenberg Hamiltonian')
    ha = nk.operator.Heisenberg(hilbert=hi, graph=g)
else:
    print('### Using the XY Hamiltonian')

if print_hamiltonian:
    try:
        print('### Hamiltonian:', ha.to_dense())
    except Exception as e:
        print('### Hamiltonian to_dense failed:', e)
        print('### This is expected if the Hamiltonian is too large to fit in memory.')


print('### Exact ground state energy')
if not exact_gs_energy:
    print('## Exact ground state energy calculation')
    try:
        evals = nk.exact.lanczos_ed(ha, compute_eigenvectors=False)
        exact_gs_energy = evals[0]
        print('N: ', N)
        print('The nk.exact calculated ground state energy: ', exact_gs_energy)
        print('The nk.exact / N calculated ground state: ', exact_gs_energy / N)
    except Exception as e:
        print('Exact ground state energy calculation failed:', e)
        print('This is expected if the Hamiltonian is too large to fit in memory.')
else:
    print('Using the provided exact ground state energy:', exact_gs_energy)

print('The analytical ground state * 4: U/N = -4/pi :', -4/jnp.pi)
print('The analytical ground state (U/N): -1/pi = ', -1/jnp.pi)
print()
print('### These should be very close if not equal and are what we are looking for from the models')
print('### Expected Ground State: ', exact_gs_energy)
print('### Expected Ground State from Analytical Solution:', -4/jnp.pi * N)
print()

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
    output_dir='out/',
    n_samples=n_s,
    n_iterations=n_i,
    seed=seed,
    alpha=alpha,
    d_max=d_m,
    learning_rate=l_r
)

rbm_model_run = ModelRun(config=rbm_config)
rbm_model_run.run()
