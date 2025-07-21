# https://netket.readthedocs.io/latest/distributed-computing.html

import jax
import netket as nk

# Initialize distributed computing
jax.distributed.initialize()

# Verify setup
print(f"[{jax.process_index()}/{jax.process_count()}] Local devices: {jax.local_devices()}", flush=True)

# Define your quantum system
L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)

# Create Hamiltonian
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)

# Define neural network model
model = nk.models.RBM(alpha=1)

# Create variational state
vs = nk.vqs.MCState(
    sampler=nk.sampler.MetropolisLocal(hi),
    model=model,
    n_samples=1024
)

# Set up optimizer and driver
opt = nk.optimizer.Sgd(learning_rate=0.01)
gs = nk.VMC(ha, opt, variational_state=vs)

# Run optimization
if jax.process_index() == 0:
    print("Starting VMC optimization...", flush=True)

gs.run(n_iter=300, out='distributed_result')

if jax.process_index() == 0:
    print("Optimization complete!", flush=True)
    print(f"Final energy: {gs.energy}", flush=True)
