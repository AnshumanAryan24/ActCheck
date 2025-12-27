import jax

# 3.3.3 -> Eq. 49
def calu(z, **kwargs):
    return z * (jax.numpy.arctan(z) / jax.numpy.pi + 0.5)
