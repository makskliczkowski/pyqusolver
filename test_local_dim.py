import jax
import jax.numpy as jnp
from QES.Solver.MonteCarlo.arsampler import ARSampler
import numpy as np

class MockNet:
    def __init__(self):
        pass
    def apply(self, variables, x, is_binary=False, method=None):
        method_str = str(method)
        if callable(method):
            res = method(self, x, is_binary)
            return res
        return jnp.zeros(x.shape[0])

    def get_phase(self, x, is_binary=False):
        return jnp.zeros(x.shape[0])

    def get_logits(self, x, is_binary=False):
        batch_size = x.shape[0]
        n_sites = x.shape[1]
        return jnp.ones((batch_size, n_sites, 3))

net = MockNet()
rng_k = jax.random.PRNGKey(0)

# Skip instantiation of ARSampler, just call the static method
print("Testing static sample ar...")
try:
    configs, log_psi = ARSampler._static_sample_ar(
        net.apply,
        {},
        rng_k,
        (4,),
        2,
        local_dim=3
    )
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
