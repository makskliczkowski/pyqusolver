import jax
import jax.numpy as jnp
from QES.Solver.MonteCarlo.arsampler import ARSampler
from QES.general_python.ml.net_impl.networks.net_autoregressive import ComplexAR
import numpy as np

try:
    net3 = ComplexAR((4,), ar_hidden=(16,), phase_hidden=(16,), local_dim=3)
    rng_k = jax.random.PRNGKey(0)
    rng = np.random.default_rng(0)

    sampler = ARSampler(
        net=net3,
        shape=(4,),
        rng_k=rng_k,
        rng=rng,
        local_dim=3,
        numsamples=2,
        numchains=1
    )

    _, res, _ = sampler.sample()
    configs, log_psi = res
    print("Sample configs shape:", configs.shape)
    print(configs)
except Exception as e:
    import traceback
    traceback.print_exc()
