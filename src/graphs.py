import jax.numpy as jnp
import netket as nk


#NOTE - These need to specify the connectivity between nodes, 
#such as X nearest neighbors or a full mesh.
#
#Ones that we need so far:
#    - 1D chain with X nearest neighbor connectivity only and optional periodic boundary conditions (pbc).
#    - Regular polygon (approximated by a circumscribed circle) with X edges (vertices) and optional center offset.
#      - This needs X nearest neighbors as well.  PBC will be handled by the X nearest neighbors.
#      - Option for full mesh connectivity (i.e., all nodes connected to all others) if needed.
#    -



def create_1d_chain_graph(nodes: int, pbc: bool = False):
    """Create a 1D chain graph with periodic boundary conditions.

    Args:
        N (int): Number of sites in the chain.
        pbc (bool): Whether to include periodic boundary conditions.

    Returns:
        nk.graph.Hypercube: A 1D chain graph.
    """
    return nk.graph.Hypercube(length=nodes, n_dim=1, pbc=pbc)

def _get_regular_polygon_nodes(num_edges: int, radius: float = 1.0, center: tuple = (0, 0)):
    """
    Calculates the coordinates of the vertices (nodes) of a regular polygon circumscribed by a circle of a given radius.

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

def create_regular_polygon_graph(num_edges: int, radius: float = 1.0, center: tuple = (0, 0), full_mesh: bool = False, basis: jnp.ndarray = jnp.array([[1, 0], [0, 1]])):
    """Create a regular polygon graph.

    Args:
        num_edges (int): Number of edges (and vertices) of the polygon.
        radius (float): Radius of the circumcircle on which the vertices lie.
        center (tuple): Center coordinates of the polygon.
        full_mesh (bool): If True, creates a full mesh connectivity between nodes.
        basis (jnp.ndarray): Basis vectors for the lattice structure.

    Returns:
        nk.graph.Graph: A regular polygon graph.
    """
    nodes = _get_regular_polygon_nodes(num_edges, radius, center)
    if nodes.size == 0:
        return None
    
    custom_edges = []
    if full_mesh:
        for i, ni in enumerate(nodes):
            for j, nj in enumerate(nodes, start=i+1):
                # Calculate the vector from node i to node j
                vector = nj - ni
                custom_edges.append((i, j, vector.tolist(), 0))  # Add color as 0 for all edges

    #g = Lattice(basis_vectors=basis, site_offsets=nodes, extent=[1, 1], pbc=False)
    return nk.graph.Lattice(basis_vectors=basis, site_offsets=nodes, extent=[1, 1], pbc=False, custom_edges=custom_edges)
    # This is another option by using a max_neighbor_order of nodes
    # g = nk.graph.Lattice(basis_vectors=basis, site_offsets=nodes, extent=[1, 1], pbc=False, max_neighbor_order=nodes)