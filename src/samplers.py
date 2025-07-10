import netket as nk

'''
This will be a collection of samplers we can use for our variational states.
NetKet provides some built-in samplers, https://netket.readthedocs.io/en/latest/api/sampler.html
'''

def create_metropolis_exchange_sampler(hi: nk.hilbert.Hilbert, g: nk.graph.Graph, d_max: int = 1):
    """
    Create a Metropolis Exchange sampler for a given Hilbert space and graph.

    Args:
        hi (nk.hilbert.Hilbert): The Hilbert space.
        g (nk.graph.Graph): The graph representing the system.
        d_max (int): Maximum distance for exchange moves.

    Returns:
        nk.sampler.MetropolisExchange: The Metropolis Exchange sampler instance.
    """
    return nk.sampler.MetropolisExchange(hilbert=hi, graph=g, d_max=d_max)


