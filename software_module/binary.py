import jax.numpy as jnp

# 3.1 -> Eq. 1
def binary(x, **kwargs):
    return jnp.where(x>=0, 1, 0)