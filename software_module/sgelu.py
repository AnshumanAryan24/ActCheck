import jax

# 3.3.2 -> Eq. 48
def sgelu(z, a=1.0, **kwargs):
    return a * z * jax.scipy.special.erf(z / jax.numpy.sqrt(2))
