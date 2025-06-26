#!/usr/local/bin/python
# -*- coding: utf-8 -*-

# Flax is a framework to define models using jax
import flax
import jax
import os
import time

# numerical operations in the model should always use jax.numpy
# instead of numpy because jax supports computing derivatives.
# If you want to better understand the difference between the two, check
# https://flax.readthedocs.io/en/latest/notebooks/jax_for_the_impatient.html
import jax.numpy as jnp
import matplotlib.pyplot as plt
import netket as nk
import scipy as sp

# Flax has two 'neural network' libraries. THe  first one is `flax.linen`
# which has been in use since 2020, and most examples use it. The new one,
# nnx, is somewhat simpler to use, and it's the one we will use here.
from flax import nnx
from netket.graph import Lattice
from netket.operator.spin import sigmax, sigmay, sigmaz
from scipy.sparse.linalg import eigsh
from matplotlib.collections import LineCollection 
from mpi4py import MPI


####### CONSTANTS - START #######
'''
Hxy = - (hbar * J / 2) SUM(i<j) 1/(r^3 of i-j) (Sx_i Sx_j + Sy_i Sy_j)
J = 2pi * 0.55 MHz
'''

N = 24 # Number of particles in the circlular chain
polygon_radius = 2.0 # the radius of the circumcircle of the polygon
basis_length = polygon_radius * 2.0 + 1.0 # must be larger than 2 * radius
polygon_center = (basis_length/2.0, basis_length/2.0)

# For future reference, the basis forms the sides of the space (cell) and 
#  site_offsets are the nodes in that cell.  As of now, nodes can be
#  considered duplicates if they would overlap another node in a DIFFERENT cell.
#  So, even when only creating 1 cell, nodes on the edges of the space can be
#  removed as duplicates because they could be duplicates if more cells are used.
# This is a long way of saying that the basis vectors need to be larger than 2 * radius
basis = jnp.array([
    [basis_length, 0.0],
    [0.0, basis_length],
])

print("basis norm: ", jnp.linalg.norm(basis, axis=1))

V = -1

# These aren't currently beint used
J = 2 * sp.constants.pi * 0.55e6  # J
# V = -J * sp.constants.hbar / 2
r = 1 # in units of nearest neighbor distance
r_meters = 16.2 * 10**-6  # in meters (actual distance between atoms)
tll_r = N * r /(2 * sp.constants.pi)
theta_N = 2 * sp.constants.pi / N  # angle between nearest neighbors in a circular chain

'''
print("tll_r: ", tll_r)
print("theta_N: ", theta_N)
print("r: ", r)
print("r_meters: ", r_meters)
print("V: ", V)
print("N: ", N)
print("J: ", J)
'''
####### CONSTANTS - END #######

class MF(nnx.Module):
    """
    A class implementing a uniform mean-field model.
    """

    # The __init__ function is used to define the parameters of the model
    # The RNG argument is used to initialize the parameters of the model.
    def __init__(self, *, rngs: nnx.Rngs):
        # To generate random numbers we need to extract the key from the
        # `rngs` object.
        key = rngs.params()
        # We store the log-wavefunction on a single site, and we call it
        # `log_phi_local`. This is a variational parameter, and it will be
        # optimized during training.
        #
        # We store a single real parameter, as we assume the wavefunction
        # is normalised, and initialise it according to a normal distibution.
        self.log_phi_local = nnx.Param(jax.random.normal(key, (1,)))

    # The __call__(self, x) function should take as
    # input a batch of states x.shape = (n_samples, L)
    # and should return a vector of n_samples log-amplitudes
    def __call__(self, x: jax.Array):

        # compute the probabilities
        p = nnx.log_sigmoid(self.log_phi_local * x)

        # sum the output
        return 0.5 * jnp.sum(p, axis=-1)
class FFN(nnx.Module):

    def __init__(self, N: int, alpha: int = 1, *, rngs: nnx.Rngs):
        """
        Construct a Feed-Forward Neural Network with a single hidden layer.

        Args:
            N: The number of input nodes (number of spins in the chain).
            alpha: The density of the hidden layer. The hidden layer will have
                N*alpha nodes.
            rngs: The random number generator seed.
        """
        self.alpha = alpha

        # We define a linear (or dense) layer with `alpha` times the number of input nodes
        # as output nodes.
        # We must pass forward the rngs object to the dense layer.
        self.linear = nnx.Linear(in_features=N, out_features=alpha * N, rngs=rngs)

    def __call__(self, x: jax.Array):

        # we apply the linear layer to the input
        y = self.linear(x)

        # the non-linearity is a simple ReLu
        y = nnx.relu(y)

        # sum the output
        return jnp.sum(y, axis=-1)

def rij(i, j):
    return jnp.sqrt(2*tll_r**2 - 2*tll_r**2*jnp.cos((i-j)*theta_N))  # Assuming a simple linear distance for demonstration
    
def distance_factor(i, j):
    # distance between 2 nodes in a circle
    return 1 / rij(i,j)**3  # Example: simple inverse square law

def get_regular_polygon_nodes(num_edges: int, radius: float = 1.0, center: tuple = (0, 0)):
    """
    Calculates the coordinates of the vertices (nodes) of a regular polygon.

    Args:
        num_edges (int): The number of edges (and vertices) of the polygon.
                         Must be an integer greater than or equal to 3.
        radius (float): The radius of the circumcircle on which the vertices lie.

    Returns:
        list: A list of tuples, where each tuple (x, y) represents the
              coordinates of a vertex.
    """
    if num_edges < 3:
        print("A polygon must have at least 3 edges.")
        return jnp.array([])

    nodes = jnp.empty((num_edges, 2), dtype=float)
    # Calculate the angle between successive vertices
    angle_increment = 2 * jnp.pi / num_edges

    for i in range(num_edges):
        # Calculate the angle for the current vertex
        angle = i * angle_increment
        # Calculate x and y coordinates
        x = radius * jnp.cos(angle) + center[0]
        y = radius * jnp.sin(angle) + center[1]
        # JAX arrays are immutable so we can't use nodes[i] = (x, y)
        nodes = nodes.at[i].set((x, y))
    return nodes

# Get the nodes
site_offset_nodes = get_regular_polygon_nodes(N, polygon_radius, polygon_center)

print(f"Nodes of a {N}-edge regular polygon with radius {polygon_radius} and center ({polygon_center[0]}, {polygon_center[1]}):")
for i, node in enumerate(site_offset_nodes):
    print(f"Node {i+1}: ({node[0]:.4f}, {node[1]:.4f})")

'''
I took this definition from the netket documentation at https://netket.readthedocs.io/en/stable/api/_generated/graph/netket.graph.Lattice.html
custom_edge =
* index of the starting point in the unit cell 
* index of the endpoint in the unit cell * vector pointing from the former to the latter 
* color of the edge (optional) If colors are not supplied, they are assigned sequentially starting from 0
example:
custom_edges = [
    (0, 0, [1.0,0.0], 0),
    (0, 0, [0.0,0.5], 1),
]
'''
custom_edges = []
for i in range(len(site_offset_nodes)):
    for j in range(i, len(site_offset_nodes)):
        if i != j:
            # Calculate the vector from node i to node j
            vector = site_offset_nodes[j] - site_offset_nodes[i]
            custom_edges.append((i, j, vector.tolist(), 0))  # Add color as 0 for all edges
            # Normalize the vector
            #norm = jnp.linalg.norm(vector)
            #if norm > 0:
                #vector /= norm
                #custom_edges.append((i, j, vector))

print("custom_edges: ", custom_edges)

#g = Lattice(basis_vectors=basis, site_offsets=site_offset_nodes, extent=[1, 1], pbc=False)
g = Lattice(basis_vectors=basis, site_offsets=site_offset_nodes, extent=[1, 1], pbc=False, custom_edges=custom_edges)
#g = Lattice(basis_vectors=basis, site_offsets=site_offset_nodes, extent=[1, 1], pbc=False, max_neighbor_order=24)
print(g.n_nodes)
print(g.basis_coords)

print(g.positions)

print("edges_count: ", g.n_edges)
print("edges: ", g.edges())
print("colors: ", g.edge_colors)

#g.draw(distance_order=23)

# check the edges and periodic boundary condition 
# each site should have three bonds

lines = []
for eg in g.edges():
    (x1,y1) = g.positions[eg[0]]
    (x2,y2) = g.positions[eg[1]]
    lines.append([(x1,y1),(x2,y2)])
    
lc = LineCollection(lines)
fig, ax = plt.subplots()
ax.add_collection(lc)
ax.autoscale()
ax.set_aspect('equal')
plt.show()

#exit()

start_time = time.time()

# hi = nk.hilbert.Spin(s=1/2, N=N)
hi = nk.hilbert.Spin(N=N, s=1/2, total_sz=0)

# There has to be a way to init H_xy to a 0 value I can add to in a for loop
#. but I don't know how so I'm setting it to the first value
print("distance factor: ", distance_factor(0,1))
H_xy = sum([V * distance_factor(0,j) * (sigmax(hi, 0) * sigmax(hi, j) + sigmay(hi, 0) * sigmay(hi, j)) for j in range(1,N)])

for i in range(1,N):
    H_i = sum([V * distance_factor(i,j) * (sigmax(hi, i) * sigmax(hi, j) + sigmay(hi, i) * sigmay(hi, j)) for j in range(i+1,N)])
    H_xy += H_i

print(H_xy)

#sp_h = H_xy.to_sparse()
#print(sp_h.shape)
#eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")

#print("eigenvalues with scipy sparse:", eig_vals)

#E_gs = eig_vals[0]
#exit()

#evals = nk.exact.lanczos_ed(H_xy, compute_eigenvectors=False)
#exact_gs_energy = evals[0]

#print('The exact ground-state energy is E0=',exact_gs_energy)
# precalculated value:
exact_gs_energy = -35.141341382447116
print('The exact ground-state energy is E0=',exact_gs_energy)
#exit()

model = FFN(N=N, alpha=1, rngs=nnx.Rngs(2))

sampler = nk.sampler.MetropolisExchange(hilbert=hi, graph=g)

# Create the local sampler on the hilbert space
#sampler = nk.sampler.MetropolisLocal(hi)

vstate = nk.vqs.MCState(sampler, model, n_samples=1008)

optimizer = nk.optimizer.Sgd(learning_rate=0.1)

# Notice the use, again of Stochastic Reconfiguration, which considerably improves the optimisation
gs = nk.driver.VMC(
    H_xy,
    optimizer,
    variational_state=vstate,
    preconditioner=nk.optimizer.SR(diag_shift=0.1),
)

log = nk.logging.RuntimeLog()
gs.run(n_iter=300, out=log)

ffn_energy = vstate.expect(H_xy)
error = -1 # I don't know yet
print("Optimized energy: ", ffn_energy)
print("--- %s seconds ---" % (time.time() - start_time))

plt.errorbar(
    log.data["Energy"].iters,
    log.data["Energy"].Mean,
    yerr=log.data["Energy"].Sigma,
    label="SymmModel",
)

plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.legend(frameon=False)
plt.show()
