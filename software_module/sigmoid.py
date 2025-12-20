import jax

# 3.2 -> Eq. 2
def sigmoid(z, **kwargs):
    return jax.nn.sigmoid(z)