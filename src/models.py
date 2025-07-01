import jax

import jax.numpy as jnp
import netket as nk

from flax import nnx

class Jastrow(nnx.Module):
    def __init__(self, N: int, *, rngs: nnx.Rngs):
        k1, k2 = jax.random.split(rngs.params())
        self.J = nnx.Param(0.01 * jax.random.normal(k1, (N, N),
                                                    dtype=jnp.complex128))

        self.v_bias = nnx.Param(0.01 * jax.random.normal(k2, (N, 1),
                                                         dtype=jnp.complex128))

    def __call__(self, x):
        x = x.astype(jnp.complex128)              # keep the dtypes aligned
        quad = jnp.einsum('...i,ij,...j->...', x, self.J.value, x)
        lin  = jnp.squeeze(x @ self.v_bias, -1)   # (...,N) @ (N,1) → (...,1)
        return quad + lin

# TODO: Rename this class
class FFModel(nnx.Module):
    def __init__(self, N: int, *, rngs: nnx.Rngs):
        k1, k2 = jax.random.split(rngs.params())
        self.J = nnx.Param(0.01 * jax.random.normal(k1, (N, N),
                                                    dtype=jnp.complex128))

        self.v_bias = nnx.Param(0.01 * jax.random.normal(k2, (N, 1),
                                                         dtype=jnp.complex128))
        self.linear = nnx.Linear(
            in_features=N, 
            out_features=2 * N, 
            dtype=jnp.complex128, 
            param_dtype=jnp.complex128,
            rngs=rngs)

    def __call__(self, x: jax.Array):
        #x = x.astype(jnp.complex128)              # keep the dtypes aligned
        x = self.linear(x)
        x = nk.nn.activation.log_cosh(x)
        x = jnp.sum(x, axis=-1)
        return x

# TODO: Rename this class    
class FFModel2(nnx.Module):
    def __init__(self, N: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(
            in_features=N, 
            out_features=2 * N, 
            dtype=jnp.complex128, 
            param_dtype=jnp.complex128,
            rngs=rngs)
        self.linear2 = nnx.Linear(
            in_features=2 * N, 
            out_features=N, 
            dtype=jnp.complex128, 
            param_dtype=jnp.complex128,
            rngs=rngs)

    def __call__(self, x: jax.Array):
        x = self.linear1(x)
        x = nk.nn.activation.log_cosh(x)
        x = self.linear2(x)
        x = nk.nn.activation.log_cosh(x)
        x = jnp.sum(x, axis=-1)
        return x
