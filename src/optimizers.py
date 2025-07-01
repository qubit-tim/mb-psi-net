import netket as nk

Note: NetKet recommends using `optax` for optimizers.

Here is the warning from their documentation
'''
Even if optimisers in netket.optimizer are optax optimisers, 
they have slightly different names (they are capitalised) and 
the argument names have been rearranged and renamed. This was 
chosen in order not to break our API from previous versions

In general, we advise you to directly use optax, as it is much 
more powerful, provides more optimisers, and it’s extremely 
easy to use step-dependent schedulers.'''

