import jax

# 3.6 -> Eq. 90
def relu(z, **kwargs):
    return jax.nn.relu(z)