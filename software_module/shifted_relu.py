import jax
import jax.numpy as jnp

# 3.6.1 -> Eq. 91
def shifted_relu(z, **kwargs):
    return jax.nn.relu(z+1)-1