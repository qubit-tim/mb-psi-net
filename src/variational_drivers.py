import optax

import netket as nk

'''
These are the drivers for the variational optimization of our models.
NetKet provides one, the Variational Monte Carlo (VMC) driver
and has a tutorial for building custom drivers:
    https://netket.readthedocs.io/en/latest/tutorials/vmc-from-scratch.html
'''

def create_vmc_driver(ha: nk.operator.DiscreteOperator, vs: nk.vqs.VariationalState, op: optax.GradientTransformationExtraArgs, sr: nk.optimizer.Preconditioner):
    """
    Create a Variational Monte Carlo (VMC) driver.

    Args:
        ha (nk.operator.Operator): The Hamiltonian operator.
        vs (nk.vqs.VariationalState): The variational state.
        op (nk.optimizer.Optimizer): The optimizer.
        sr (nk.optimizer.Preconditioner): The preconditioner.

    Returns:
        nk.VMC: The VMC driver instance.
    """
    return nk.VMC(
        hamiltonian=ha,
        optimizer=op,
        preconditioner=sr,
        variational_state=vs
    )
