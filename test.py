import jax
import jax.numpy as jnp

x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])
print("x + y =", x + y)

@jax.jit
def f(z):
    return z * z + 2

print("f(x) =", f(x))

def g(z):
    return jnp.sum(z ** 2)

grad_g = jax.grad(g)
print("grad g(x) =", grad_g(x))

key = jax.random.key(0)
samples = jax.random.normal(key, shape=(3,))
print("random samples =", samples)

