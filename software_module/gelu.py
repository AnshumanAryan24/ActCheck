import jax

# 3.3.1 -> Eq. 45
def gelu(z, approximate=False, **kwargs):
    return jax.nn.gelu(z, approximate=approximate)
