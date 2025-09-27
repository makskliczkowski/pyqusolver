'''
Handles the network implementations and utilities for Neural Quantum States (NQS).
'''
from nqs_backend import *

#########################################
#! Gradients and network utilities
#########################################

def nqs_choose_network(net_spec         : Any,
                        input_shape     : tuple,
                        backend         : BackendInterface,
                        **kwargs
                    ) -> Networks.GeneralNet:
    """
    Select, instantiate, and initialize a network object for NQS.
    Parameters:
    -----------
    net_spec : Any
        Network specification: can be a string (network type), a Flax module, or a GeneralNet instance.
    input_shape : tuple
        Shape of the input data (e.g., (num_spins,) for spin systems).
    backend : BackendInterface
        Backend instance (e.g., NumpyBackend, JaxBackend).
    **kwargs : dict
        Additional keyword arguments for network initialization.
    Returns:
    --------
    Networks.GeneralNet
        An initialized network object ready for use.
    """

    # Case 1: already a GeneralNet
    if isinstance(net_spec, Networks.GeneralNet):
        if not net_spec.initialized:
            net_spec.init()
        return net_spec

    # Case 2: Flax module or FlaxInterface
    if (JAX_AVAILABLE and "flax" in globals() and (isinstance(net_spec, nn.Module) or isinstance(net_spec, Networks.FlaxInterface))):
        if backend.name != "jax":
            raise ValueError(f"Flax module {net_spec} requires JAX backend, got {backend.name}")
        
        net = Networks.FlaxInterface(net_module=net_spec, input_shape=input_shape, backend=backend.name, **kwargs)
        
        if not net.initialized:
            net.init()
        return net

    # Case 3: string (factory lookup)
    if isinstance(net_spec, str):
        net = Networks.choose_network(network_type=net_spec, input_shape=input_shape, backend=backend.name, **kwargs)

        if not net.initialized:
            net.init()
        return net

    # Case 4: unsupported
    raise TypeError(f"Unsupported network specification type: {type(net_spec)}")

# ----------------------------------------------
#! EOF
# ----------------------------------------------