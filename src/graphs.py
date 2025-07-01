import jax.numpy as jnp
import netket as nk
from traitlets import This


NOTE - These need to specify the connectivity between nodes, 
such as X nearest neighbors or a full mesh.

Ones that we need so far:
    - 1D chain with X nearest neighbor connectivity only and optional periodic boundary conditions (pbc).
    - Regular polygon (approximated by a circumscribed circle) with X edges (vertices) and optional center offset.
      - This needs X nearest neighbors as well.  PBC will be handled by the X nearest neighbors.
      - Option for full mesh connectivity (i.e., all nodes connected to all others) if needed.
    -

def create_1D_chain_graph(N: int, pbc: bool = False):
    """Create a 1D  chain graph with periodic boundary conditions.

    Args:
        N (int): Number of sites in the chain.
        pbc (bool): Whether to include periodic boundary conditions.

    Returns:
        nk.graph.Hypercube: A 1D  chain graph.
    """
    return nk.graph.Hypercube(length=N, n_dim=1, pbc=pbc)

def _get_regular_polygon_nodes(num_edges: int, radius: float = 1.0, center: tuple = (0, 0)):
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

def create_regular_polygon_graph(num_edges: int, radius: float = 1.0, center: tuple = (0, 0)):
    """Create a regular polygon graph.

    Args:
        num_edges (int): Number of edges (and vertices) of the polygon.
        radius (float): Radius of the circumcircle on which the vertices lie.
        center (tuple): Center coordinates of the polygon.

    Returns:
        nk.graph.Graph: A regular polygon graph.
    """
    nodes = _get_regular_polygon_nodes(num_edges, radius, center)
    if nodes.size == 0:
        return None

    # Create edges between consecutive nodes and wrap around for periodicity
    edges = [(i, (i + 1) % num_edges) for i in range(num_edges)]
    
    # Create the graph using NetKet's Graph class
    return nk.graph.Graph(edges=edges, nodes=nodes)