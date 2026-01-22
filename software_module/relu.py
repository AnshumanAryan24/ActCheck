import jax
import jax.numpy as jnp

# 3.6 -> Eq. 90
def relu(z, **kwargs):
    return jax.nn.relu(z)


# 3.6.1 -> Eq. 91
def shifted_relu(z, **kwargs):
    return jnp.maximum(z, -1.0)


# 3.6.2 -> Eq. 92
def lrelu(z, a=100.0, **kwargs):
    return jnp.where(z >= 0, z, z / a)


# 3.6.2 (Very Leaky ReLU) -> described on page 16
def vlrelu(z, a=3.0, **kwargs):
    return jnp.where(z >= 0, z, z / a)


# 3.6.2 (Optimized Leaky ReLU) -> Eq. 93-94
def olrelu(z, l=3.0, u=8.0, a=None, **kwargs):
    if a is None:
        a = (u + l) / (u - l)
    return jnp.where(z >= 0, z, z * jnp.exp(-a))


# 3.6.3 -> Eq. 95-96
def rrelu(z, l=3.0, u=8.0, key=None, training=False, **kwargs):
    if training:
        if key is None:
            raise ValueError("rrelu requires a PRNG key when training=True.")
        a = jax.random.uniform(key, shape=jnp.shape(z), minval=l, maxval=u)
    else:
        a = (l + u) / 2.0
    return jnp.where(z >= 0, z, z / a)


# 3.6.4 -> Eq. 97
def srrelu(z, l=0.125, u=1.0 / 3.0, key=None, training=False, **kwargs):
    if training:
        if key is None:
            raise ValueError("srrelu requires a PRNG key when training=True.")
        a = jax.random.uniform(key, shape=jnp.shape(z), minval=l, maxval=u)
    else:
        a = (l + u) / 2.0
    softsign_term = 1.0 / jnp.square(1.0 + z)
    return softsign_term + jnp.where(z >= 0, z, a * z)


# 3.6.5 -> Eq. 98
def slrelu(z, a=1.0, **kwargs):
    return jnp.where(z >= 0, a * z, 0.0)


# 3.6.6 -> Eq. 99
def nrelu(z, sigma=None, key=None, training=False, **kwargs):
    if training:
        if key is None:
            raise ValueError("nrelu requires a PRNG key when training=True.")
        if sigma is None:
            sigma = jnp.std(z)
        noise = jax.random.normal(key, shape=jnp.shape(z)) * sigma
    else:
        noise = 0.0
    return jnp.maximum(0.0, z + noise)


# 3.6.7 -> Eq. 100
def sinerelu(z, a=1.0, **kwargs):
    return jnp.where(z >= 0, z, a * (jnp.sin(z) - jnp.cos(z)))


# 3.6.8 -> Eq. 101
def minsin(z, **kwargs):
    return jnp.minimum(z, jnp.sin(z))


# 3.6.9 -> Eq. 102
def vlu(z, a=1.0, b=1.0, **kwargs):
    return jnp.maximum(0.0, z) + a * jnp.sin(b * z)


# 3.6.10 -> Eq. 104
def scaa(x, depthwise_fn=None, **kwargs):
    if depthwise_fn is None:
        raise ValueError("scaa requires depthwise_fn to compute fDW(X).")
    return jnp.maximum(x, depthwise_fn(x))


# 3.6.11 -> Eq. 105
def rt_relu(z, sigma=0.75, key=None, training=False, **kwargs):
    if training:
        if key is None:
            raise ValueError("rt_relu requires a PRNG key when training=True.")
        jitter = jax.random.normal(key, shape=jnp.shape(z)) * sigma
    else:
        jitter = 0.0
    return jnp.maximum(0.0, z + jitter)


# 3.6.12 -> Eq. 106
def nlrelu(z, a=1.0, **kwargs):
    return jnp.log1p(a * jnp.maximum(0.0, z))


# 3.6.13 -> Eq. 108
def slu(z, **kwargs):
    neg = 2.0 * jax.nn.softplus(z) - 2.0 * jnp.log(2.0)
    return jnp.where(z >= 0, z, neg)


# 3.6.14 -> Eq. 109
def resp(z, a=1.5, **kwargs):
    return jnp.where(z >= 0, a * z + jnp.log(2.0), jax.nn.softplus(z))


# 3.6.15 -> Eq. 110
def prenu(z, a=1.0, **kwargs):
    pos = z - a * jnp.log1p(jnp.maximum(z, 0.0))
    return jnp.where(z >= 0, pos, 0.0)


# 3.6.16 -> Eq. 111
def brelu(z, a=6.0, **kwargs):
    return jnp.clip(z, 0.0, a)


# 3.6.17 -> Eq. 112
def hard_sigmoid(z, **kwargs):
    return jnp.maximum(0.0, jnp.minimum((z + 1.0) / 2.0, 1.0))


# 3.6.18 -> Eq. 114
def hard_tanh(z, a=-1.0, b=1.0, **kwargs):
    return jnp.clip(z, a, b)


# 3.6.19 -> Eq. 115
def sv_hardtanh(z, a=0.0, **kwargs):
    return jnp.where(z < -1.0, -1.0 + a, jnp.where(z > 1.0, 1.0 + a, z + a))


# 3.6.19 -> Eq. 117
def sh_hardtanh(z, a=0.0, **kwargs):
    return jnp.where(z < -1.0 - a, -1.0, jnp.where(z > 1.0 - a, 1.0, z))


# 3.6.20 -> Eq. 118
def hard_swish(z, **kwargs):
    return z * jnp.clip(z / 6.0 + 0.5, 0.0, 1.0)
