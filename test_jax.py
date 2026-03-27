import jax
import jax.numpy as jnp

logits = jnp.array([[1.0, 2.0], [3.0, 1.0]])
rng_key = jax.random.PRNGKey(0)

# sample categorical
sample = jax.random.categorical(rng_key, logits)
print(sample)

log_probs = jax.nn.log_softmax(logits, axis=-1)
print(log_probs)
