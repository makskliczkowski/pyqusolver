import jax
import jax.numpy as jnp

rng = jax.random.PRNGKey(0)
logits = jnp.array([[-10.0, 10.0], [10.0, -10.0]])
gumbel_noise = jax.random.gumbel(rng, logits.shape)
noisy_logits = logits + gumbel_noise
samples = jnp.argmax(noisy_logits, axis=-1)
print(samples)
