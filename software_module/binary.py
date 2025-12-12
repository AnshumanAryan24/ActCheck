import jax.numpy as jnp

def binary(x, **kwargs):
    return jnp.where(x>=0, 1, 0)