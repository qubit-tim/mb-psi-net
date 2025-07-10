import netket as nk

from flax import nnx

'''
These are the variational states, the heart of NetKet itself.
I am not sure we will need to define our own or even need this file
I will leave it here for now and list out the ones we do use:
    - MCState - Variational State for a Variational Neural Quantum State.
    - MCMixedState - Variational State for a Mixed Variational Neural Quantum State.
      - This was the AI generated part, is it right? Variational State for a mixed state, i.e., a density matrix
'''

def create_mc_variational_state(sa: nk.sampler.Sampler, ma: nnx.Module, n_samples: int = 1000):
    """
    Create a Monte Carlo (MC) variational state for a given sampler and machine.

    Args:
        sa (nk.sampler.Sampler): The sampler to use.
        ma (nk.machine.Machine): The machine to use.
        n_samples (int): Number of samples to draw.

    Returns:
        nk.vqs.MCState: The Monte Carlo state instance.
    """
    return nk.vqs.MCState(sa, ma, n_samples=n_samples)

