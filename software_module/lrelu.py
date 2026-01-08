import jax

# 3.6.2 -> Eq. 92
def lrelu(z, a=0.01, **kwargs):
    return jax.nn.leaky_relu(z, a)