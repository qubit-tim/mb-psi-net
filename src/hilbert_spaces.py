import netket as nk

'''
I am not sure we need this yet but if we have complicated Hilbert spaces, this might be nice
for now though, I will list out the ones we use:
    - Spin 1/2, N sites, total Sz = 0 or Z
    - 
    
'''

def create_spin_half_hilbert_space(g: nk.graph.Graph, total_sz: int = 0):
    """
    Create a Spin 1/2 Hilbert space for N sites with total Sz = 0.

    Args:
        g (nk.graph.Graph): The graph representing the system.

    Returns:
        nk.hilbert.Spin: The Spin 1/2 Hilbert space.
    """
    # Assuming g.n_nodes gives the number of sites
    return nk.hilbert.Spin(s=0.5, total_sz=total_sz, N=g.n_nodes)

