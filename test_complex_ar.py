import jax
import jax.numpy as jnp
from QES.general_python.ml.net_impl.networks.net_autoregressive import ComplexAR

# Test local_dim=2
net = ComplexAR((4,), ar_hidden=(16,), phase_hidden=(16,), local_dim=2)
x = jnp.array([[0.5, -0.5, 0.5, -0.5], [-0.5, 0.5, -0.5, 0.5]])
params = net.get_params()
log_psi = net._flax_module.apply({'params': params}, x)
print("log_psi shape (dim=2):", log_psi.shape)

# For methods on FlaxInterface
# Need to see how they map, but net._flax_module is the inner object
logits = net._flax_module.apply({'params': params}, x, method=lambda m,x: m.get_logits(x))
print("logits shape (dim=2):", logits.shape)

# Test local_dim=3
net3 = ComplexAR((4,), ar_hidden=(16,), phase_hidden=(16,), local_dim=3)
x3 = jnp.array([[0, 1, 2, 0], [2, 1, 0, 2]])
params3 = net3.get_params()
log_psi3 = net3._flax_module.apply({'params': params3}, x3)
print("log_psi shape (dim=3):", log_psi3.shape)

logits3 = net3._flax_module.apply({'params': params3}, x3, method=lambda m,x: m.get_logits(x))
print("logits shape (dim=3):", logits3.shape)
