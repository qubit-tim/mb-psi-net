import jax.numpy as jnp
import netket as nk

from netket.operator.spin import sigmax, sigmay, sigmaz

# 1D XY Model Hamiltonian (check the paper for details)
def create_1d_xy_hamiltonian(hi: nk.hilbert.Hilbert, g: nk.graph.Graph, pbc: bool = False):
    """
    Create a 1D XY Hamiltonian for N sites with optional periodic boundary conditions (pbc).

    Args:
        N (int): Number of sites in the chain.
        pbc (bool): Whether to include periodic boundary conditions.
        heisenberg_hamiltonian (bool): Whether to use the Heisenberg Hamiltonian instead of the XY Hamiltonian.

    Returns:
        nk.operator.Operator: The XY Hamiltonian operator.
    """
    N = g.n_nodes
    ha = sum([(sigmax(hi, i) * sigmax(hi, i+1 if i+1 < N else 0) + sigmay(hi, i) * sigmay(hi, i+1 if i+1 < N else 0)) for i in range(0,N)])
    if not pbc:
        ha = sum([(sigmax(hi, i) * sigmax(hi, i+1) + sigmay(hi, i) * sigmay(hi, i+1)) for i in range(0,N-1)])
    
    return ha

def create_heisenberg_hamiltonian(hi: nk.hilbert.Hilbert, g: nk.graph.Graph):
    """
    Create a 1D Heisenberg Hamiltonian for N sites with optional periodic boundary conditions (pbc).

    Args:
        N (int): Number of sites in the chain.
        pbc (bool): Whether to include periodic boundary conditions.

    Returns:
        nk.operator.Operator: The Heisenberg Hamiltonian operator.
    """
    return nk.operator.Heisenberg(hilbert=hi, graph=g)

def regular_polygon_node_distance(i, j, N, r=1):
    # distance between 2 nodes in a circle
    theta_N = 2 * jnp.pi / N
    tll_r = N * r /(2 * jnp.pi)
    rij = jnp.sqrt(2*tll_r**2 - 2*tll_r**2*jnp.cos((i-j)*theta_N))
    return 1 / rij**3  


# TLL Hamiltonian
def create_tll_hamiltonian(hi: nk.hilbert.Hilbert, g: nk.graph.Graph):
    """
    Create a Tomonaga-Luttinger Liquid (TLL) Hamiltonian for N sites with optional periodic boundary conditions (pbc).

    Args:
        N (int): Number of sites in the chain.

    Returns:
        nk.operator.Operator: The TLL Hamiltonian operator.
    """
    # There has to be a way to init H_xy to a 0 value I can add to in a for loop
    #. but I don't know how so I'm setting it to the first value
    print("distance factor: ", regular_polygon_node_distance(0,1,N=6))
    N = g.n_nodes
    H_xy = sum([regular_polygon_node_distance(0,j,N=6) * (sigmax(hi, 0) * sigmax(hi, j) + sigmay(hi, 0) * sigmay(hi, j)) for j in range(1,N)])

    for i in range(1,N):
        H_i = sum([regular_polygon_node_distance(i,j,N=6) * (sigmax(hi, i) * sigmax(hi, j) + sigmay(hi, i) * sigmay(hi, j)) for j in range(i+1,N)])
        H_xy += H_i

    return H_xy