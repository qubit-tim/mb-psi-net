import jax
import json
import os
import time
import warnings
import random

import jax.numpy as jnp
import matplotlib.pyplot as plt
import netket as nk

from flax import nnx
from mpi4py import MPI
from netket.operator.spin import sigmax, sigmay, sigmaz

# CONSTANTS
output_dir = 'out/'

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

def JastrowRun(hilbert, graph, hamiltonian, exact_sol=0, n_samples=1000, n_iterations=1000, rngs=0, d_max=1, learning_rate=0.01, holomorphic=False):
    print('### Jastrow calculation')
    hi = hilbert
    g = graph
    ha = hamiltonian
    # Model
    ma = Jastrow(N=hi.size, rngs=nnx.Rngs(rngs))
    # Sampler
    sa = nk.sampler.MetropolisExchange(hilbert=hi, graph=g, d_max=d_max)
    # Optimizer
    op = nk.optimizer.Sgd(learning_rate=learning_rate)
    # Stochastic Reconfiguration
    sr = nk.optimizer.SR(diag_shift=0.1, holomorphic=holomorphic)
    # Variational state
    vs = nk.vqs.MCState(sa, ma, n_samples=n_samples)
    # Ground-state optimization loop
    gs = nk.VMC(
        hamiltonian=ha,
        optimizer=op,
        preconditioner=sr,
        variational_state=vs)

    start = time.time()
    out = str(output_dir) + 'Jastrow' + '-it-' + str(n_iterations) + '-sa-' + str(n_samples) + '-sd-' + str(rngs)
    gs.run(out=out, n_iter=n_iterations)
    end = time.time()
    
    print('### Jastrow calculation finished')
    print('### Jastrow Results')
    print('- Ground state energy approximation: ', gs.energy)
    if exact_sol != 0:
        error = jnp.abs((gs.energy.mean - exact_sol) / exact_sol)
        print('- Relative Error: ', error)
        print('- Percentage Error: ', error * 100, '%')
    print('- Number of Parameters: ',nk.jax.tree_size(vs.parameters))
    print('- Seconds to perform the calculation:',end-start)
    print()

def JastrowGraph(exact_sol=0, logfile="Jastrow.log"):
    if not os.path.exists(logfile):
        print(f"The file '{logfile}' does not exist.")
        print("Please run the JastrowRun function first to generate the log file.\n")
        return
    data_Jastrow=json.load(open(logfile, 'r'))
    
    iters_Jastrow = data_Jastrow["Energy"]["iters"]
    energy_Jastrow = data_Jastrow["Energy"]["Mean"]["real"]

    fig, ax1 = plt.subplots()
    ax1.plot(iters_Jastrow, energy_Jastrow, color='C8', label='Energy (Jastrow)')
    ax1.set_ylabel('Energy')
    ax1.set_xlabel('Iteration')
    #plt.axis([0,iters_Jastrow[-1],exact_sol-0.1,exact_sol+1.0])
    if not exact_sol == 0:
        plt.axhline(y=exact_sol, xmin=0,
                    xmax=iters_Jastrow[-1], linewidth=2, color='k', label='Exact')
    ax1.legend()
    plt.show()

def RBMRun(hilbert, graph, hamiltonian, exact_sol=0, n_samples=1000, n_iterations=1000, rngs=0, alpha=1, d_max=1, learning_rate=0.01, holomorphic=False):
    print('### RBM calculation')
    hi = hilbert
    g = graph
    ha = hamiltonian
    # alpha => ratio of visible spins to virtual spins in upper layers
    # Model
    ma = nk.models.RBM(alpha=alpha, param_dtype=jnp.complex128)
    # Sampler
    sa = nk.sampler.MetropolisExchange(hilbert=hi,graph=g, d_max=d_max)
    # Optimizer
    op = nk.optimizer.Sgd(learning_rate=learning_rate)
    # Stochastic Reconfiguration
    sr = nk.optimizer.SR(diag_shift=0.1, holomorphic=holomorphic)
    # Variational State
    vs = nk.vqs.MCState(sa, ma, n_samples=n_samples)
    # ground-state optimization loop
    gs = nk.VMC(
        hamiltonian=ha,
        optimizer=op,
        preconditioner=sr,
        variational_state=vs)

    start = time.time()
    out = str(output_dir) + 'RBM' + '-it-' + str(n_iterations) + '-sa-' + str(n_samples) + '-sd-' + str(rngs)
    gs.run(out=out, n_iter=n_iterations)
    end = time.time()

    print('### RBM calculation finished')
    print('### RBM Results')
    print('- Ground state energy approximation: ', gs.energy)
    if exact_sol != 0:
        error = jnp.abs((gs.energy.mean - exact_sol) / exact_sol)
        print('- Relative Error: ', error)
        print('- Percentage Error: ', error * 100, '%')
    print('- Number of Parameters: ',nk.jax.tree_size(vs.parameters))
    print('- Seconds to perform the calculation:',end-start)
    print()

def RBMGraph(exact_sol=0, logfile="RBM.log"):
    if not os.path.exists(logfile):
        print(f"The file '{logfile}' does not exist.")
        print("Please run the RBMRun function first to generate the log file.\n")
        return
    data_RBM=json.load(open(logfile, 'r'))
    
    iters_RBM = data_RBM["Energy"]["iters"]
    energy_RBM = data_RBM["Energy"]["Mean"]["real"]

    fig, ax1 = plt.subplots()
    ax1.plot(iters_RBM, energy_RBM, color='C8', label='Energy (RBM)')
    ax1.set_ylabel('Energy')
    ax1.set_xlabel('Iteration')
    #plt.axis([0,iters_RBM[-1],exact_sol-0.1,exact_sol+1.0])
    if not exact_sol == 0:
        plt.axhline(y=exact_sol, xmin=0,
                    xmax=iters_RBM[-1], linewidth=2, color='k', label='Exact')
    ax1.legend()
    plt.show()

def RBMSymmRun(hilbert, graph, hamiltonian, exact_sol=0, n_samples=1000, n_iterations=1000, rngs=0, alpha=1, d_max=1, learning_rate=0.01, holomorphic=False):
    print('### Symmetric RBM calculation')
    hi = hilbert
    g = graph
    ha = hamiltonian
    # Model
    ma = nk.models.RBMSymm(symmetries=g.translation_group(), alpha=alpha, param_dtype=jnp.complex128)
    # Sampler
    sa = nk.sampler.MetropolisExchange(hi, graph=g, d_max=d_max)
    # Optimizer
    op = nk.optimizer.Sgd(learning_rate=learning_rate)
    # Stochastic Reconfiguration
    sr = nk.optimizer.SR(diag_shift=0.1, holomorphic=holomorphic)
    # The variational state
    vs = nk.vqs.MCState(sa, ma, n_samples=n_samples)
    # The ground-state optimization loop
    gs = nk.VMC(
        hamiltonian=ha,
        optimizer=op,
        preconditioner=sr,
        variational_state=vs)

    start = time.time()
    out = str(output_dir) + 'RBMSymmetric' + '-it-' + str(n_iterations) + '-sa-' + str(n_samples) + '-sd-' + str(rngs)
    gs.run(out=out, n_iter=n_iterations)
    end = time.time()
    
    print('### Symmetric RBM calculation finished')
    print('### Symmetric RBM Results')
    print('- Ground state energy approximation: ', gs.energy)
    if exact_sol != 0:
        error = jnp.abs((gs.energy.mean - exact_sol) / exact_sol)
        print('- Relative Error: ', error)
        print('- Percentage Error: ', error * 100, '%')
    print('- Number of Parameters: ',nk.jax.tree_size(vs.parameters))
    print('- Seconds to perform the calculation:',end-start)
    print()

def RBMSymmGraph(exact_sol=0, logfile="RBMSymmetric.log"):
    if not os.path.exists(logfile):
        print(f"The file '{logfile}' does not exist.")
        print("Please run the RBMSymmRun function first to generate the log file.\n")
        return
    data_RBMSymm=json.load(open(logfile, 'r'))
    
    iters_RBMSymm = data_RBMSymm["Energy"]["iters"]
    energy_RBMSymm = data_RBMSymm["Energy"]["Mean"]["real"]

    fig, ax1 = plt.subplots()
    ax1.plot(iters_RBMSymm, energy_RBMSymm, color='C8', label='Energy (RBM Symmetric)')
    ax1.set_ylabel('Energy')
    ax1.set_xlabel('Iteration')
    #plt.axis([0,iters_RBMSymm[-1],exact_sol-0.1,exact_sol+1.0])
    if not exact_sol == 0:
        plt.axhline(y=exact_sol, xmin=0,
                    xmax=iters_RBMSymm[-1], linewidth=2, color='k', label='Exact')
    ax1.legend()
    plt.show()

def FFRun(hilbert, graph, hamiltonian, exact_sol=0, n_samples=1000, n_iterations=1000, rngs=0, d_max=1, learning_rate=0.01, holomorphic=False):
    print('### Feed Forward calculation')
    hi = hilbert
    g = graph
    ha = hamiltonian
    # Model
    ffnn = FFModel(N=hi.size, rngs=nnx.Rngs(rngs))
    # Sampler
    sa = nk.sampler.MetropolisExchange(hi, graph=g, d_max=d_max)
    # Variational State
    vs = nk.vqs.MCState(sa, ffnn, n_samples=n_samples)
    # Optimizer
    op = nk.optimizer.Sgd(learning_rate=learning_rate)
    # Stochastic Reconfiguration
    sr = nk.optimizer.SR(diag_shift=0.1, holomorphic=holomorphic)
    # Ground-state optimization loop
    gs = nk.VMC(
        hamiltonian=ha,
        optimizer=op,
        preconditioner=sr,
        variational_state=vs)


    start = time.time()
    out = str(output_dir) + 'FF' + '-it-' + str(n_iterations) + '-sa-' + str(n_samples) + '-sd-' + str(rngs)
    gs.run(out=out, n_iter=n_iterations)
    end = time.time()
    
    print('### Feed Forward calculation finished')
    print('### Feed Forward Results')
    print('- Ground state energy approximation: ', gs.energy)
    if exact_sol != 0:
        error = jnp.abs((gs.energy.mean - exact_sol) / exact_sol)
        print('- Relative Error: ', error)
        print('- Percentage Error: ', error * 100, '%')
    print('- Number of Parameters: ',nk.jax.tree_size(vs.parameters))
    print('- Seconds to perform the calculation:',end-start)
    print()

def FFGraph(exact_sol=0, logfile="FF.log"):
    if not os.path.exists(logfile):
        print(f"The file '{logfile}' does not exist.")
        print("Please run the FFRun function first to generate the log file.\n")
        return
    data_FF=json.load(open(logfile, 'r'))
    
    iters_FF = data_FF["Energy"]["iters"]
    energy_FF = data_FF["Energy"]["Mean"]["real"]

    fig, ax1 = plt.subplots()
    ax1.plot(iters_FF, energy_FF, color='C8', label='Energy (Feed Forward)')
    ax1.set_ylabel('Energy')
    ax1.set_xlabel('Iteration')
    #plt.axis([0,iters_FF[-1],exact_sol-0.1,exact_sol+1.0])
    if not exact_sol == 0:
        plt.axhline(y=exact_sol, xmin=0,
                    xmax=iters_FF[-1], linewidth=2, color='k', label='Exact')
    ax1.legend()
    plt.show()

def FF2Run(hilbert, graph, hamiltonian, exact_sol=0, n_samples=1000, n_iterations=1000, rngs=0, d_max=1, learning_rate=0.01, holomorphic=False):
    print('### Feed Forward 2 calculation')
    hi = hilbert
    g = graph
    ha = hamiltonian
    # Model
    ffnn = FFModel2(N=hi.size, rngs=nnx.Rngs(rngs))
    # Sampler
    sa = nk.sampler.MetropolisExchange(hi, graph=g, d_max=d_max)
    # Variational State
    vs = nk.vqs.MCState(sa, ffnn, n_samples=n_samples)
    # Optimizer
    op = nk.optimizer.Sgd(learning_rate=learning_rate)
    # Stochastic Reconfiguration
    sr = nk.optimizer.SR(diag_shift=0.1, holomorphic=holomorphic)
    # Ground-state optimization loop
    gs = nk.VMC(
        hamiltonian=ha,
        optimizer=op,
        preconditioner=sr,
        variational_state=vs)

    start = time.time()
    out = str(output_dir) + 'FF2' + '-it-' + str(n_iterations) + '-sa-' + str(n_samples) + '-sd-' + str(rngs)
    gs.run(out=out, n_iter=n_iterations)
    end = time.time()
    
    print('### Feed Forward 2 calculation finished')
    print('### Feed Forward 2 Results')
    print('- Ground state energy approximation: ', gs.energy)
    if exact_sol != 0:
        error = jnp.abs((gs.energy.mean - exact_sol) / exact_sol)
        print('- Relative Error: ', error)
        print('- Percentage Error: ', error * 100, '%')
    print('- Number of Parameters: ',nk.jax.tree_size(vs.parameters))
    print('- Seconds to perform the calculation:',end-start)
    print()

def FF2Graph(exact_sol=0, logfile="FF2.log"):
    if not os.path.exists(logfile):
        print(f"The file '{logfile}' does not exist.")
        print("Please run the FF2Run function first to generate the log file.\n")
        return
    data_FF2=json.load(open(logfile, 'r'))
    
    iters_FF2 = data_FF2["Energy"]["iters"]
    energy_FF2 = data_FF2["Energy"]["Mean"]["real"]

    fig, ax1 = plt.subplots()
    ax1.plot(iters_FF2, energy_FF2, color='C8', label='Energy (Feed Forward 2)')
    ax1.set_ylabel('Energy')
    ax1.set_xlabel('Iteration')
    #plt.axis([0,iters_FF2[-1],exact_sol-0.1,exact_sol+1.0])
    if not exact_sol == 0:
        plt.axhline(y=exact_sol, xmin=0,
                    xmax=iters_FF2[-1], linewidth=2, color='k', label='Exact')
    ax1.legend()
    plt.show()

# Set up MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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


JastrowRun(hi, g, ha, exact_sol=exact_gs_energy, n_samples=n_s, n_iterations=n_i, rngs=seed, d_max=d_m, learning_rate=l_r)
RBMRun(hi, g, ha, exact_sol=exact_gs_energy, n_samples=n_s, n_iterations=n_i, rngs=seed, alpha=1, d_max=d_m)
RBMSymmRun(hi, g, ha, exact_sol=exact_gs_energy, n_samples=n_s, n_iterations=n_i, rngs=seed, alpha=1, d_max=d_m)
FFRun(hi, g, ha, exact_sol=exact_gs_energy, n_samples=n_s, n_iterations=n_i, rngs=seed, d_max=d_m, learning_rate=l_r)
FF2Run(hi, g, ha, exact_sol=exact_gs_energy, n_samples=n_s, n_iterations=n_i, rngs=seed, d_max=d_m, learning_rate=l_r)
