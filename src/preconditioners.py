import netket as nk

'''
This will be a collection of preconditioners we could use for our optimizers.
NetKet provides one, Stochastic Reconfiguration or Natural Gradient preconditioner for the gradient,
'''

def create_sr_preconditioner(holomorphic: bool = False):
    """
    Create a Stochastic Reconfiguration (SR) preconditioner.

    Args:
        holomorphic (bool): Whether to use a holomorphic SR preconditioner.

    Returns:
        nk.optimizer.Preconditioner: The SR preconditioner instance.
    """
    return nk.optimizer.SR(diag_shift=0.1, holomorphic=holomorphic)