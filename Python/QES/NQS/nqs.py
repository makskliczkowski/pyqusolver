"""
QES.NQS.nqs
===========

Neural Quantum State (NQS) Solver.

Implements a Monte Carlo-based training method for optimizing NQS models.
Supports both NumPy and JAX backends for efficiency and flexibility.

Usage
-----
Import and use the NQS solver:

    from QES.NQS.nqs import NQS
    nqs = NQS(...)

See the documentation and examples for details.
----------------------------------------------------------
Author          : Maks Kliczkowski
Email           : maxgrom97@gmail.com
Date            : 01.10.25
Description     : Neural Quantum State (NQS) Solver implementation.
----------------------------------------------------------
"""

import json
import h5py
import warnings

# typing and other imports
from typing import Union, Tuple, Union, Callable, Optional, Any, Sequence, List, Dict
from functools import partial
from pathlib import Path
from dataclasses import dataclass

# Import timeit utility for timing code blocks
try:
    from QES.general_python.common.timer import timeit
except ImportError as e:
    raise ImportError("Failed to import timer module. Ensure QES.general_python is correctly installed.") from e

# import physical problems
try:
    from .src.nqs_physics import *
    from .src.nqs_network_integration import *
    from .src.nqs_engine import NQSEvalEngine
except ImportError as e:
    raise ImportError("Failed to import nqs_physics or nqs_networks module. Ensure QES.NQS is correctly installed.") from e

# ----------------------------------------------------------

# from QES.general_python imports
try:
    #! Algebra
    from QES.Algebra.Operator.operator import Operator, OperatorFunction
    
    #! Randomness
    from QES.general_python.common.directories import Directories
    from QES.general_python.common.flog import Logger
    
    #! Monte Carlo
    from QES.Solver.MonteCarlo.vmc import VMCSampler, get_sampler
    from QES.Solver.MonteCarlo.montecarlo import MonteCarloSolver
    
    #! Hilbert space
    from QES.Algebra.hilbert import HilbertSpace
    
    #! Choose network
    from QES.general_python.ml.networks import choose_network as nqs_choose_network
    
except ImportError as e:
    warnings.warn("Some QES.general_python modules could not be imported. Ensure QES.general_python is installed correctly.", ImportWarning)
    raise e

# ----------------------------------------------------------
AnsatzFunctionType      = Callable[[Callable, Array, int, Any], Array]               # func, states, batch_size, params -> Array [log ansatz values]
ApplyFunctionType       = Callable[[Callable, Array, Array, Array, int, Any], Array] # func, states, probs, params, batch_size -> Array [sampled/evaluated values]

#########################################
#! NQS main class
#########################################

@dataclass
class NQSSingleStepResult:
    '''
    Data class to hold the results of a single optimization step in the NQS solver.
    '''
    loss                : float                     # Estimated energy or loss value
    loss_mean           : float                     # Mean of the energy or loss
    loss_std            : float                     # Standard deviation of the energy or loss
    grad_flat           : Array                     # Flattened gradient vector
    params_shapes       : List[Tuple]               # Shapes of the parameters
    params_sizes        : List[int]                 # Sizes of the parameters
    params_cpx          : List[bool]                # Whether each parameter is complex

#########################################

class NQS(MonteCarloSolver):
    '''
    Neural Quantum State (NQS) Solver.
    
    Implements a Monte Carlo-based training method for optimizing NQS models.
    Supports both NumPy and JAX backends for efficiency and flexibility.
    '''
    
    _ERROR_ALL_DTYPE_SAME       = "All weights must have the same dtype!"
    _ERROR_NOT_INITIALIZED      = "The NQS network is not initialized. Call init_network() first."
    _ERROR_INVALID_PHYSICS      = "Invalid physics problem specified."
    _ERROR_NO_HAMILTONIAN       = "Hamiltonian must be provided for wavefunction physics."
    _ERROR_INVALID_HAMILTONIAN  = "Invalid Hamiltonian type provided."
    _ERROR_INVALID_NETWORK      = "Invalid network type provided."
    _ERROR_INVALID_SAMPLER      = "Invalid sampler type provided."
    _ERROR_INVALID_BATCH_SIZE   = "Batch size must be a positive integer."
    _ERROR_STATES_PSI           = "If providing states and psi, both must be provided as a tuple (states, psi)."
    
    def __init__(self,
                # information on the NQS
                net         : Union[Callable, str, net_general.GeneralNet],
                sampler     : Union[Callable, str, VMCSampler],
                model       : Hamiltonian,
                # information on the Monte Carlo solver
                batch_size  : Optional[int]                             = 1,
                *,
                # information on the NQS
                nparticles  : Optional[int]                             = None,
                # information on the Monte Carlo solver     
                seed        : Optional[int]                             = None,
                beta        : float                                     = 1,
                mu          : float                                     = 2.0,
                replica     : int                                       = 1,
                # information on the NQS - Hilbert space
                shape       : Union[list, tuple]                        = (1,),
                hilbert     : Optional[Union[HilbertSpace, list, tuple]] = None,
                modes       : int                                       = 2,
                # information on the Monte Carlo solver
                directory   : Optional[str]                             = MonteCarloSolver.defdir,
                backend     : str                                       = 'default',
                problem     : Optional[Union[str, PhysicsInterface]]    = 'wavefunction',
                **kwargs):
        '''
        Initialize the NQS solver.
        
        Parameters:
            net:
                The neural network to be used. This can be specified in several ways:
                - As a string (e.g., 'rbm', 'cnn') to use a built-in network.
                - As a pre-initialized network object (must be a subclass of `GeneralNet`).
                - As a raw Flax module class for custom networks. The factory will wrap it automatically.
                  See the documentation of `QES.general_python.ml.networks.choose_network` for details
                  on custom module requirements.
            sampler:
                The sampler to be used.
            model:
                The physical model (e.g., Hamiltonian) for the NQS.
                - If physics is 'wavefunction', this must provide a `local_energy_fn`.
                - For other physics types, refer to the specific requirements.
            batch_size:
                The batch size for training.
            nparticles [Optional]:
                The number of particles in the system. If not provided, defaults to system size.
            seed:
                Random seed for initialization.
            beta:
                Inverse temperature for Monte Carlo sampling.
            mu:
                Mu parameter for the NQS.
            replica:
                Number of replicas for parallel tempering.
            shape:
                Shape of the input data, e.g., `(n_spins,)`.
            hilbert:
                Hilbert space object (optional).
            modes:
                Number of local modes per site (e.g., 2 for spin-1/2).
            directory:
                Directory for saving results.
            backend:
                Computational backend ('jax' or 'numpy').
            problem:
                The physics problem to solve (e.g., 'wavefunction').
            **kwargs:
                Additional keyword arguments passed to the network constructor.
        '''
        super().__init__(sampler    =   sampler,
                        seed        =   seed,
                        beta        =   beta,
                        mu          =   mu,
                        replica     =   replica,
                        shape       =   shape,
                        hilbert     =   hilbert,
                        modes       =   modes,
                        directory   =   directory,
                        backend     =   backend)
        
        # --------------------------------------------------
        self._batch_size            = batch_size        
        self._initialized           = False
        
        self._modifier              = None #! State modifier (for later), for quench dynamics, etc.
        self._modifier_func         = None

        #######################################
        #! collect the Hilbert space information
        #######################################
        
        self._nh                    = self._hilbert.Nh if self._hilbert is not None else None
        self._nparticles            = nparticles if nparticles is not None else self._size
        self._nvisible              = self._size
        self._nparticles2           = self._nparticles**2
        self._nvisible2             = self._nvisible**2
        self._beta_penalty          = kwargs.get('beta_penalty', 0.0)
        
        # --------------------------------------------------
        #! Backend
        # --------------------------------------------------
        self._nqsbackend            = nqs_get_backend(self._backend_str)                # from src/nqs_backend.py
        
        # --------------------------------------------------
        #! Physical problem to solve
        # --------------------------------------------------
        problem                     = PhysicsInterfaces[problem.upper()] if isinstance(problem, str) else problem
        self._nqsproblem            = nqs_choose_physics(problem, self._nqsbackend)     # from src/nqs_physics.py
        
        # --------------------------------------------------
        #! Network 
        # --------------------------------------------------
        
        self._net                   = nqs_choose_network(net, input_shape=self._shape, backend=self._nqsbackend, **kwargs)
        if not self._initialized:
            self.init_network()
        
        self._isjax                 = getattr(self._net, 'is_jax', (self._nqsbackend.name == "jax"))
        self._iscpx                 = self._net.is_complex
        self._holomorphic           = self._net.is_holomorphic
        self._analytic              = self._net.has_analytic_grad
        self._dtype                 = self._net.dtype
        
        # --------------------------------------------------
        #! Handle gradients
        # --------------------------------------------------
        self._grad_info             = self._nqsbackend.prepare_gradients(self._net)    
        self._flat_grad_func        = self._grad_info["flat_grad_func"]         # Function to compute flattened gradients
        self._dict_grad_type        = self._grad_info["dict_grad_type"]         # Dictionary of gradient types
        self._params_slice_metadata = self._grad_info["slice_metadata"]         # Metadata for slicing parameters
        self._params_leaf_info      = self._grad_info["leaf_info"]              # Leaf information
        self._params_tree_def       = self._grad_info["tree_def"]               # Tree definition
        self._params_shapes         = self._grad_info["shapes"]                 # Shapes of parameters
        self._params_sizes          = self._grad_info["sizes"]                  # Sizes of parameters
        self._params_iscpx          = self._grad_info["is_complex_per_leaf"]    # Whether each parameter is complex
        self._params_total_size     = self._grad_info["total_size"]             # Total size of parameters -> cost for training
        
        # --------------------------------------------------
        #! Compile functions
        # --------------------------------------------------
        self._ansatz_func, self._eval_func, self._apply_func = self.nqsbackend.compile_functions(self._net, batch_size=self._batch_size)

        # --------------------------------------------------
        #! Model and physics setup
        # --------------------------------------------------

        self._model                 = model
        self._nqsproblem.setup(self._model, self._net)
        # For wavefunction problem we keep the same attribute name you used:
        if self._nqsproblem.typ == 'wavefunction':
            self._local_en_func = getattr(self._nqsproblem, "local_energy_fn", None)
        else:
            raise ValueError(self._ERROR_INVALID_PHYSICS)

        # --------------------------------------------------
        #! Initialize unified evaluation engine
        # --------------------------------------------------
        if self._nqsproblem.typ == 'wavefunction':
            self._eval_engine = NQSEvalEngine(self, backend='auto', batch_size=batch_size, jit_compile=True, **kwargs)
        else:
            raise ValueError(self._ERROR_INVALID_PHYSICS)

        #######################################
        #! Directory to save the results
        #######################################
        self._init_directory()
        self._dir.mkdir()
        self._dir_detailed.mkdir()
        
        #! Orbax checkpoint manager
        self._use_orbax             = kwargs.get('use_orbax', True)
        self._orbax_max_to_keep     = kwargs.get('orbax_max_to_keep', 3)

        if (self._isjax and self._use_orbax and hasattr(self._net, 'net_module') and isinstance(self._net.net_module, nn.Module)):

            #! set the checkpoint manager
            ckpt_base = Path(self._dir_detailed / 'checkpoints')
            ckpt_base.mkdir(parents=True, exist_ok=True)
            
            handler   = PyTreeCheckpointHandler()
            self._ckpt_manager = CheckpointManager(
                directory           = str(ckpt_base),
                checkpoint_type     = handler,
                max_to_keep         = self._orbax_max_to_keep)
        else:
            self._ckpt_manager = None
    
    #####################################
    #! INITIALIZATION OF THE NETWORK AND FUNCTIONS
    #####################################
    
    def reset(self):
        """
        Resets the initialization state of the object and reinitializes the underlying network.
        This method marks the object as not initialized and forces a reinitialization of the associated
        neural network by calling its `force_init` method.
        """
        
        self._initialized = False
        self._net.force_init()
    
    # ---
    
    def _init_directory(self):
        """
        Initializes the directory for saving results.
        This method creates a directory structure based on the problem model and network configuration.
        It ensures that the directory exists and is ready for use.
        """
        
        base                = Directories(self._dir)
        detailed            = base.join(str(self._model), create=False)

        #! add the lattice information if needed
        if self._model.lattice is not None:
            detailed        = detailed.join(str(self._model.lattice), create=False)
        else:
            detailed        = detailed.join(str(self._nvisible), create=False)

        #! network summary: skip sampler & params
        #   e.g. "FlaxNetInterface_in12_dtypecomplex128"
        
        net_cls             = type(self._net).__name__
        try:
            dim             = self._net.input_dim
            dtype           = getattr(self._net, "dtype", self._dtype)
            seed            = getattr(self._net, "seed", None)
        except AttributeError:
            dim, dtype = None, None

        parts               = [net_cls]
        if dim   is not None: parts.append(f"in={dim}")
        if dtype is not None: parts.append(f"dtype={dtype}")
        if seed  is not None: parts.append(f"seed={seed}")

        net_folder          = "_".join(parts)
        final_dir           = detailed.join(net_folder, create=False)

        #! actually mkdir them all
        #    create intermediate parents automatically
        base.mkdir()
        final_dir.mkdir()

        #! store for later use
        self._dir           = base
        self._dir_detailed  = final_dir
    
    # ---
    
    def init_network(self, forced: bool = False):
        '''
        Initialize the network truly. This means that the weights are initialized correctly 
        and the dtypes are checked. In addition, the network is checked if it is holomorphic or not.
        Parameters:
            s: The state vector, can be any, but it is used to initialize the network.
        
        Note:
            1. Check if the network is already initialized.
            2. If not, initialize the weights using the network's init method.
            3. Check the dtypes of the weights and ensure they are consistent.
            4. Check if the network is complex and holomorphic.
            5. Check the shape of the weights and store them.
            6. Calculate the number of parameters in the network.
            7. Set the initialized flag to True.
            8. If the network is not initialized, raise a ValueError.
        '''

        if not self._initialized or forced:
            
            # initialize the network
            self._params            = self._net.init(self._rng_k)
            dtypes                  = self._net.dtypes
            
            # check if all dtypes are the same
            if not all([a == dtypes[0] for a in dtypes]):
                raise ValueError(self._ERROR_ALL_DTYPE_SAME)
            
            # check if the network is complex
            self._iscpx             = not (dtypes[0] == np.single or dtypes[0] == np.double)
            
            # check if the network is holomorphic
            # if the value is set to None, we check if the network is holomorphic
            # through calculating the gradients of the real and imaginary parts
            # of the network. Otherwise, we use the value provided.
            self._holomorphic       = self._net.check_holomorphic()
            self.log(f"Network is holomorphic: {self._holomorphic}", log='info', lvl = 2, color = 'blue')
            self._analytic          = self._net.has_analytic_grad
            self.log(f"Network has analytic gradient: {self._analytic}", log='info', lvl = 2, color = 'blue')
            
            # check the shape of the weights
            self._paramshape        = self._net.shapes
            
            # number of parameters
            self._nparams           = self._net.nparams
            self._initialized       = True
    
    #####################################
    #! SETTERS FOR HELP
    #####################################
    
    def _set_batch_size(self, batch_size: int):
        '''
        Set the batch size for the network evaluation.
        This method updates the batch size used for evaluating the network and recompiles.
        Parameters:
            batch_size: The new batch size to set.
        '''
        
        if batch_size is None or self._batch_size == batch_size:
            return
        self._batch_size = batch_size
        self._ansatz_func, self._eval_func, self._apply_func = self.nqsbackend.compile_functions(self._net, batch_size=self._batch_size)
    
    #####################################
    #! EVALUATION OF THE ANSATZ BATCHED (\psi(s))
    #####################################
    
    def evaluate(self, states, batch_size=None, params=None):
        '''
        Evaluate the neural network (log ansatz) for the given quantum states.
        
        This uses the unified evaluation engine for efficient computation with both
        JAX and NumPy backends, and automatic batching for memory efficiency.
        
        Parameters:
            states      : The state configurations to evaluate.
            batch_size  : The size of batches to use for the evaluation.
            params      : The parameters (weights) to use for the network evaluation.
        Returns:
            The evaluated network output (log ansatz values).
        '''
        result = self._eval_engine.evaluate_ansatz(states, params, batch_size)
        return result.values
    
    def ansatz(self, states, batch_size=None, params=None): 
        ''' Alias for the log ansatz evaluation '''
        return self.evaluate(states, batch_size, params)
    
    def __call__(self, states):
        '''
        Evaluate the network using the provided state. This
        will return the log ansatz of the state coefficient. Uses
        the default backend for this class - using self._eval_func.
        
        Parameters:
            states:
                The state vector.
        Returns:
            The evaluated network output.
        '''
        return self._eval_func(func=self._ansatz_func, data=states, batch_size=self._batch_size, params=self.get_params())
    
    @property
    def eval_func(self):
        '''
        Returns the evaluation function for the network.
        This function is used to evaluate the network on a batch of states.
        We assume it returns log(psi(s)).
        
        Returns:
            Callable: The evaluation function.
        '''
        return self._eval_func
    
    @property
    def eval_fun(self):      
        'Alias for eval_func. We assume it returns log(psi(s))'
        return self._eval_func
    @property
    def ansatz_func(self):      
        'Alias for ansatz_func. We assume it returns log(psi(s))'
        return self._ansatz_func
    @property
    def ansatz_fun(self):  
        'Alias for ansatz_func. We assume it returns log(psi(s))'
        return self._ansatz_func
    @property
    def log_ansatz_func(self):  
        'Alias for ansatz_func. We assume it returns log(psi(s))'
        return self._ansatz_func
    
    #####################################
    #! APPLY FUNCTION VALUES - LOCAL ENERGY AND OTHER FUNCTIONS (OPERATORS)
    #####################################
    
    @staticmethod
    @partial(jax_jit, static_argnames=['func', 'logproba_fun', 'batch_size'])
    def _apply_fun_jax(func   : Callable,                           # function to be evaluated (e.g., local energy f: s -> E_loc(s))
                states        : Array,                              # input states (shape: (N, ...))
                probabilities : Array,                              # probabilities associated with the states (shape: (N,))
                logproba_in   : Array,                              # logarithm of the probabilities for the input states (\log p(s))
                logproba_fun  : Callable,                           # function to compute the logarithm of probabilities (\log p(s') -> to evaluate)
                parameters    : Union[dict, list, jnp.ndarray],     # parameters to be passed to the function - for the Networks ansatze
                batch_size    : Optional[int] = None):              # batch size for evaluation
        """
        Evaluates a given function on a set of states and probabilities, with optional batching.
        Args:
            func (Callable):
                The function to be evaluated.
            states (jnp.ndarray):
                The input states for the function.
            probabilities (jnp.ndarray):
                The probabilities associated with the states.
            logproba_in (jnp.ndarray):
                The logarithm of the probabilities for the input states.
            logproba_fun (Callable):
                A function to compute the logarithm of probabilities.
            parameters (Union[dict, list, jnp.ndarray]):
                Parameters to be passed to the function.
            batch_size (Optional[int], optional):
                The size of batches for evaluation. 
                If None, the function is evaluated without batching. Defaults to None.
        Returns:
            The result of the function evaluation, either batched or unbatched, depending on the value of `batch_size`.
        """
        if batch_size is None or batch_size == 1:
            funct_in = net_utils.jaxpy.apply_callable_jax
            return funct_in(func, states, probabilities, logproba_in, logproba_fun, parameters)
        else:
            funct_in = net_utils.jaxpy.apply_callable_batched_jax
            return funct_in(func            = func,
                            states          = states,
                            sample_probas   = probabilities,
                            logprobas_in    = logproba_in,
                            logproba_fun    = logproba_fun,
                            parameters      = parameters,
                            batch_size      = batch_size)
    
    @staticmethod
    def _apply_fun_np(func    : Callable,
                states        : np.ndarray,
                probabilities : np.ndarray,
                logproba_in   : np.ndarray,
                logproba_fun  : Callable,
                parameters    : Union[dict, list, np.ndarray],
                batch_size    : Optional[int] = None):
        """
        Evaluates a given function on a set of states and probabilities, with optional batching.
        
        Args:
            func (Callable):
                The function to be evaluated.
            states (np.ndarray):
                The input states for the function.
            probabilities (np.ndarray):
                The probabilities associated with the states.
            logproba_in (np.ndarray):
                The logarithm of the probabilities for the input states.
            logproba_fun (Callable):
                A function to compute the logarithm of probabilities.
            parameters (Union[dict, list, np.ndarray]):
                Parameters to be passed to the function.
            batch_size (Optional[int], optional):
                The size of batches for evaluation. 
                If None, the function is evaluated without batching. Defaults to None.
        Returns:
            The result of the function evaluation, either batched or unbatched, depending on the value of `batch_size`.
        """
        
        if batch_size is None:
            funct_in = net_utils.numpy.apply_callable_np # (func, states, probabilities, logproba_in, logproba_fun, parameters)
            return funct_in(func, states, probabilities, logproba_in, logproba_fun, parameters)
        
        # otherwise, we shall use the batched version
        funct_in = net_utils.numpy.apply_callable_batched_np # (func, states, sample_probas, logprobas_in, logproba_fun, parameters, batch_size)
        return funct_in(func            = func,
                        states          = states,
                        sample_probas   = probabilities,
                        logprobas_in    = logproba_in,
                        logproba_fun    = logproba_fun,
                        parameters      = parameters,
                        batch_size      = batch_size)

    @staticmethod
    def _apply_fun(func     : Callable,
            states          : np.ndarray,
            probabilities   : np.ndarray,
            logproba_in     : np.ndarray,
            logproba_fun    : Callable,
            parameters      : Union[dict, list, np.ndarray],
            batch_size      : Optional[int] = None,
            is_jax          : bool = True):
        """
        Evaluates a given function on a set of states and probabilities, with optional batching.
        
        Args:
            func (Callable):
                The function to be evaluated.
            states (Union[np.ndarray, jnp.ndarray]):
                The input states for the function.
            probabilities (Union[np.ndarray, jnp.ndarray]):
                The probabilities associated with the states.
            logproba_in (Union[np.ndarray, jnp.ndarray]):
                The logarithm of the probabilities for the input states.
            logproba_fun (Callable):
                A function to compute the logarithm of probabilities.
            parameters (Union[dict, list, np.ndarray]):
                Parameters to be passed to the function.
            batch_size (Optional[int], optional):
                The size of batches for evaluation. 
                If None, the function is evaluated without batching. Defaults to None.
            is_jax (bool, optional):
                Flag indicating if JAX is used for computation. Defaults to True.
        Returns:
            The result of the function evaluation, either batched or unbatched, depending on the value of `batch_size`.
        """
                    
        if batch_size is None:
            funct_in = net_utils.jaxpy.apply_callable_jax if is_jax else net_utils.numpy.apply_callable_np
            return funct_in(func, states, probabilities, logproba_in, logproba_fun, parameters)
        
        # otherwise, we shall use the batched version
        funct_in = net_utils.jaxpy.apply_callable_batched_jax if is_jax else net_utils.numpy.apply_callable_batched_np
        return funct_in(func            = func,
                        states          = states,
                        sample_probas   = probabilities,
                        logprobas_in    = logproba_in,
                        logproba_fun    = logproba_fun,
                        parameters      = parameters,
                        batch_size      = batch_size)
    
    @staticmethod
    def _apply_fun_s(func   : list[Callable],
                sampler     : VMCSampler,
                num_samples : int,
                num_chains  : int,
                logproba_fun: Callable,
                parameters  : dict,
                batch_size  : Optional[int] = None,
                is_jax      : bool = True):
        """
        Evaluates a given function using samples generated by a sampler.

        This method utilizes a sampler to generate states, ansatze, and their 
        associated probabilities, and then evaluates the provided function 
        using these samples.

        Args:
            func (Callable)                 : The function to be evaluated. It should accept 
                                            states, probabilities, ansatze, logproba_fun, and parameters 
                                            as inputs.
            sampler (Sampler)               : The sampler object used to generate samples.
            num_samples (int)               : The total number of samples to generate.
            num_chains (int)                : The number of independent Markov chains to use 
                                            in the sampling process.
            logproba_fun (Callable)         : A function that computes the logarithm 
                                            of the probability for given states.
            parameters (dict)               : A dictionary of parameters to be passed to 
                                            the function being evaluated.
            batch_size (Optional[int])      : The size of batches to process at a 
                                            time. If None, the entire dataset is processed at once.
                                            Defaults to None.
            is_jax (bool, optional)         : Flag indicating if JAX is used for computation. 
                                            Defaults to True.
        Returns:    
            Any: The result of evaluating the provided function `func` using 
            the generated samples.
        """

        _, (states, ansatze), probabilities = sampler.sample(parameters=parameters, num_samples=num_samples, num_chains=num_chains)
        evaluated_results = [NQS._apply_fun(f, states, probabilities, ansatze, logproba_fun, parameters, batch_size, is_jax) for f in func]
        return (states, ansatze), probabilities, evaluated_results
    
    def apply(self,
            functions       : Optional[Union[List, Callable]]                                                   = None,
            states_and_psi  : Optional[Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray]]]   = None,
            probabilities   : Optional[Union[np.ndarray, jnp.ndarray]]                                          = None,
            batch_size      : Optional[int]                                                                     = None,
            **kwargs):
        """
        Evaluate a set of functions based on the provided states, wavefunction, and probabilities.
        This method computes the output of one or more functions using the provided states, 
        wavefunction (ansatze), and probabilities. If states and wavefunction are not provided, 
        it uses a sampler to generate the required data.
        
        Args:
            functions (Optional[list]):
                A list of functions to evaluate. If not provided, 
                defaults to using the local energy function (`self._local_en_func`).
            states_and_psi (Optional[Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray]]]):
                A tuple containing the states and the corresponding wavefunction (ansatze). 
                If not provided, the sampler is used to generate these.
            probabilities (Optional[Union[np.ndarray, jnp.ndarray]]):
                Probabilities associated with the states. If not provided, defaults to an array of ones with the same 
                shape as the wavefunction ansatze.
            **kwargs:
                Additional keyword arguments:
                    - batch_size (int)          : The batch size for evaluation. Defaults to self._batch_size.
                    - num_samples (int)         : Number of samples to generate if using the sampler.
                    - num_chains (int)          : Number of chains to use if using the sampler.
        Returns:
            Union[Any, list]:
                The output of the evaluated functions. If a single function is 
                provided, the result is returned directly. If multiple functions are provided, 
                a list of results is returned.
        """
        
        params          = kwargs.get('params', self._net.get_params())
        batch_size      = batch_size if batch_size is not None else self._batch_size
        
        #! check if the functions are provided
        if functions is None or (isinstance(functions, list) and len(functions) == 0):
            functions = [self._local_en_func]
        elif not isinstance(functions, list):
            functions = [functions]
        
        # check if the states and psi are provided
        states, ansatze = None, None
        if states_and_psi is not None:
            if isinstance(states_and_psi, tuple) and len(states_and_psi) == 2:
                states, ansatze = states_and_psi
            elif isinstance(states_and_psi, np.ndarray) or isinstance(states_and_psi, jnp.ndarray):
                states          = states_and_psi    # assume it's the states
                ansatze         = self(states)      # call log ansatz function
            else:
                raise ValueError(self._ERROR_STATES_PSI)
        else:
            # get other parameters from kwargs
            num_samples = kwargs.get('num_samples', self._sampler.numsamples)
            num_chains  = kwargs.get('num_chains', self._sampler.numchains)
            _, (states, ansatze), probabilities = self._sampler.sample(parameters=params, num_samples=num_samples, num_chains=num_chains)
            
        # check if the probabilities are provided
        if probabilities is None:
            probabilities = self._backend.ones_like(ansatze).astype(ansatze.dtype)
            
        output = [
            self._apply_func(
                func            = f,
                states          = states,
                sample_probas   = probabilities,
                logprobas_in    = ansatze,
                logproba_fun    = self._ansatz_func,
                parameters      = params,
                batch_size      = batch_size
            ) for f in functions
        ]
        
        # check if the output is a list
        if isinstance(output, list) and len(output) == 1:
            output = output[0]
        return (states, ansatze), probabilities, output

    def __getitem__(self, funct: Union[Callable, List[Callable]]):
        """
        Allows the object to be indexed using square brackets with a callable or a list of callables.
        When accessed, applies the given function(s) to the object using the `apply` method.

        Args:
            funct (Union[Callable, List[Callable]]): A single callable or a list of callables to be applied.

        Returns:
            The result of applying the given function(s) to the object.

        Raises:
            Any exception raised by the `apply` method.
        """

        return self.apply(funct)
    
    @property
    def apply_func(self):
        '''
        Returns the apply function for the network.
        This function is used to apply a function to a batch of states.
        
        Returns:
            Callable: The apply function.
        '''
        return self._apply_fun_jax if self._isjax else self._apply_fun_np
    
    @property
    def apply_fun(self):    
        'Alias for apply_func'    
        return self.apply_func
    @property
    def apply_function(self): 
        'Alias for apply_func' 
        return self.apply_func
    
    #####################################
    #! UNIFIED EVALUATION INTERFACE USING EVALUATION ENGINE - USER FRIENDLY
    #####################################

    def compute_energy(self, states, ham_action_func=None, params=None, probabilities=None, batch_size=None):
        """
        Compute local energies using the evaluation engine.
        
        Parameters:
        -----------
            states: 
                Array of state configurations
            ham_action_func: 
                Function computing local energy
            params: 
                Optional network parameters
            probabilities: 
                Optional probability weights
            batch_size: 
                Optional batch size override

        Returns:
            EnergyStatistics object with energy values and statistics
        """
        if ham_action_func is None:     ham_action_func = self._local_en_func
        if params is None:              params          = self._net.get_params()
        if batch_size is None:          batch_size      = self._batch_size
        
        if self._nqsproblem.typ == 'wavefunction':
            return self._eval_engine.compute_local_energy(states, ham_action_func, params, probabilities, batch_size)
        raise NotImplementedError("Energy computation is only implemented for wavefunction problems.")
    
    def compute_observable(self, observable_func, states, observable_name="Observable", params=None, compute_expectation=False, batch_size=None):
        """
        Evaluate an observable using the evaluation engine.
        
        Parameters:
        -----------
            observable_func: 
                Function computing observable values
            states: 
                Array of state configurations
            observable_name: 
                Name of the observable
            params: 
                Optional network parameters. If not provided, uses current network parameters.
            compute_expectation: 
                Whether to compute expectation value
            batch_size: 
                Optional batch size override

        Returns:
            ObservableResult with local values and statistics
        """
        if observable_func is None:         raise ValueError("Observable function must be provided.")
        if params is None:                  params          = self._net.get_params()
        if batch_size is None:              batch_size      = self._batch_size
        return self._eval_engine.compute_observable(observable_func, states, observable_name, params, compute_expectation, batch_size)
        
    @property
    def eval_engine(self):
        """Get the evaluation engine for advanced use cases."""
        return self._eval_engine
    
    def energy(self, *args, **kwargs):          return self.compute_energy(*args, **kwargs)
    def local_energy(self, *args, **kwargs):    return self.compute_energy(*args, **kwargs)
    def local_en(self, *args, **kwargs):        return self.compute_energy(*args, **kwargs)
    def en(self, *args, **kwargs):              return self.compute_energy(*args, **kwargs)
    def observable(self, *args, **kwargs):      return self.compute_observable(*args, **kwargs)
    def obs(self, *args, **kwargs):             return self.compute_observable(*args, **kwargs)
    def compute_obs(self, *args, **kwargs):     return self.compute_observable(*args, **kwargs)
    
    #####################################
    #! SAMPLE
    #####################################
    
    def sample(self, num_samples = None, num_chains = None, reset: bool = False, *, params = None, **kwargs):
        '''
        Sample the NQS using the provided sampler. This will return
        the sampled states and the corresponding probabilities.
        Parameters:
            num_samples (int):
                The number of samples to generate.
            num_chains (int):
                The number of chains to use for sampling.
            reset (bool):
                Whether to reset the sampler before sampling. This 
                corresponds to reinitializing the state before sampling the new ones.
            params (dict):
                The parameters (weights) to use for the network evaluation.
                If None, uses the current parameters stored in network._params.
            kwargs:
                Additional arguments for the sampler.
        Returns:
            The sampled states and the corresponding probabilities.
            (last configs, last ansatze), (all configs, all ansatze), (all probabilities)
            
        Example:
        ----------
            >>> nqs = NQS(model, net, sampler)
            >>> (states, ansatze), probabilities = nqs.sample(num_samples=1000, num_chains=10)
            >>> print("Sampled states:", states)
            >>> # Sampled states: [[...], [...], ...]
            >>> print("Sampled ansatze:", ansatze)
            >>> # Sampled ansatze: [[...], [...], ...]
            >>> print("Sampled probabilities:", probabilities)
            >>> # Sampled probabilities: [[...], [...], ...]        
        '''
        if reset and hasattr(self._sampler, 'reset'):
            self._sampler.reset()
        
        if params is None:
            params = self._net.get_params()
            
        return self._sampler.sample(parameters=params, num_samples=num_samples, num_chains=num_chains, **kwargs)
    
    @property
    def sampler(self):
        '''
        Returns the sampler used for sampling the NQS.
        '''
        return self._sampler
    
    @property
    def sampler_func(self, num_samples: int = None, num_chains: int = None):
        '''
        Returns the sampler function used for sampling the NQS.
        '''
        return self._sampler.get_sampler(num_samples=num_samples, num_chains=num_chains)

    #####################################
    #! GRADIENTS AND LOG DERIVATIVES - FOR OPTIMIZATION
    #####################################
    
    @staticmethod
    @partial(jax.jit, static_argnums=(0, 3, 4))
    def log_derivative_jax(
            net_apply                   : Callable,                                     # The network's apply function f(p, x)
            params                      : Any,                                          # Network parameters p
            states                      : jnp.ndarray,                                  # Input states s_i, shape (num_samples, ...)
            single_sample_flat_grad_fun : Callable[[Callable, Any, Any], jnp.ndarray],  # JAX-traceable function computing the flattened gradient for one sample.
            batch_size                  : int = 1) -> Array:                            # Batch size   
        '''
        Compute the batch of flattened gradients using JAX (JIT compiled).

        Returns the gradients (e.g., :math:`O_k = \\nabla \\ln \\psi(s)`)
        for each state in the batch. The output format (complex/real)
        depends on the `single_sample_flat_grad_fun` used.

        Parameters
        ----------
        net_apply : Callable
            The network's apply function `f(p, x)`. Static argument for JIT.
        params : Any
            Network parameters `p`.
        states : jnp.ndarray
            Input states `s_i`, shape `(num_samples, ...)`.
        single_sample_flat_grad_fun : Callable[[Callable, Any, Any], jnp.ndarray]
            JAX-traceable function computing the flattened gradient for one sample.
            Signature: `fun(net_apply, params, single_state) -> flat_gradient_vector`.
            Static argument for JIT.
        batch_size : int
            Batch size. Static argument for JIT.

        Returns
        -------
        jnp.ndarray
            Array of flattened gradients, shape `(num_samples, num_flat_params)`.
            Dtype matches the output of `single_sample_flat_grad_fun`.
        
        Example:
            >>> nqs = NQS(model, net, sampler)
            >>> (states, ansatze), probabilities = nqs.sample(num_samples=1000, num_chains=10)
            >>> print("Sampled states:", states)
            >>> print("Sampled ansatze:", ansatze)
            >>> print("Sampled probabilities:", probabilities)
            >>> gradients, shapes, sizes, is_cpx = nqs.log_derivative_jax(
            ...     net_apply=nqs._ansatz_func,
            ...     params=nqs._net.get_params(),
            ...     states=states,
            ...     single_sample_flat_grad_fun=nqs._flat_grad_func,
            ...     batch_size=64
            ... )
            >>> print("Computed gradients:", gradients)
            >>> # Computed gradients: [[...], [...], ...]
        '''
        
        # The dtype (complex/real) depends on single_sample_flat_grad_fun
        gradients_batch, shapes, sizes, is_cpx = net_utils.jaxpy.compute_gradients_batched(net_apply, params, states, single_sample_flat_grad_fun, batch_size)
        return gradients_batch, shapes, sizes, is_cpx
    
    @staticmethod
    def log_derivative_np(net, params, batch_size, states, flat_grad) -> np.ndarray:
        r'''
        !TODO: Add the precomputed gradient vector - memory efficient
        '''
        sb = net_utils.numpy.create_batches_np(states, batch_size)
        
        # compute the gradients using NumPy's loop
        g = np.zeros((len(sb),) + params.shape[1:], dtype=np.float64)
        for i, b in enumerate(sb):
            g[i] = flat_grad(net, params, b)
        return g
    
    def log_derivative(self, states, batch_size = None, params = None, *args, **kwargs) -> Array:
        r'''
        Compute the gradients of the ansatz logarithmic wave-function using JAX or NumPy.
        We assume that the log ansatz function is used here and therefore it 
        computes :math:`O_k = \\partial_{W_k} \\log \\psi(s) = \\partial_{W_k} \\psi(s) / \\psi(s)`.
        
        Parameters
        ----------
            states: 
                The state vector.
            batch_size: 
                The size of batches to use for the evaluation.
            params: 
                The parameters (weights) to use for the network evaluation.
        Returns
        -------
            The computed gradients.
            
        Example:
        ----------
            >>> nqs = NQS(model, net, sampler)
            >>> (states, ansatze), probabilities = nqs.sample(num_samples=1000, num_chains=10)
            >>> print("Sampled states:", states)
            >>> # Sampled states: [[...], [...], ...]
            >>> print("Sampled ansatze:", ansatze)
            >>> # Sampled ansatze: [[...], [...], ...]
            >>> print("Sampled probabilities:", probabilities)
            >>> # Sampled probabilities: [[...], [...], ...]
            >>> gradients, shapes, sizes, is_cpx = nqs.log_derivative(
            ...     states=states,
            ...     batch_size=64,
            ...     params=nqs._net.get_params()
            ... )
            >>> print("Computed gradients:", gradients)
            >>> # Computed gradients: [[...], [...], ...]
        '''
        
        # check if the batch size is provided
        batch_size = batch_size if batch_size is not None else self._batch_size
        if batch_size is None:
            batch_size = 1
            
        # check if the parameters are provided
        params = self._net.get_params() if params is None else params
        
        if self._isjax:
            if not self._analytic:
                return self.log_derivative_jax(self._ansatz_func, params, states, self._flat_grad_func, batch_size)
            return self.log_derivative_jax(self._grad_func, params, states, self._flat_grad_func, batch_size)
        
        if not self._analytic:
            return self.log_derivative_np(self._ansatz_func, params, batch_size, states, self._flat_grad_func)
        return self.log_derivative_np(self._grad_func, params, batch_size, states, self._flat_grad_func)

    def log_deriv(self, *args, **kwargs):
        r''' Alias for log_derivative. \partial _W log(psi) = \partial _W psi / psi '''
        return self.log_derivative(*args, **kwargs)
    def log_gradient(self, *args, **kwargs):       
        r''' Alias for log_derivative. \partial _W log(psi) = \partial _W psi / psi '''
        return self.log_derivative(*args, **kwargs)
    def log_grad(self, *args, **kwargs):           
        r''' Alias for log_derivative. \partial _W log(psi) = \partial _W psi / psi '''
        return self.log_derivative(*args, **kwargs)
    
    @property
    def log_derivative_func(self):
        '''
        Returns the gradient function for the network.
        This function is used to compute the gradients of the network.
        
        Returns:
            Callable: The gradient function.
        '''
        return self.log_derivative_jax if self._isjax else self.log_derivative_np

    @property
    def log_deriv_func(self):                       return self.log_derivative_func
    @property
    def log_gradient_func(self):                    return self.log_derivative_func
    @property
    def log_grad_func(self):                        return self.log_derivative_func
    
    @property
    def flat_grad(self):
        ''' Return the flat gradient function. '''
        return self._flat_grad_func
    @property
    def flat_grad_func(self):
        ''' Return the flat gradient function. '''
        return self._flat_grad_func
    @property
    def flat_gradient_func(self):
        ''' Return the flat gradient function. '''
        return self._flat_grad_func

    #####################################
    #! UPDATE PARAMETERS
    #####################################
    
    def transform_flat_params(self,
                            flat_params : jnp.ndarray,
                            shapes      : list,
                            sizes       : list,
                            is_cpx      : bool) -> Any:
        """
        Transform a flat parameter vector into a PyTree structure.
        This is important function as it allows to convert the flat
        parameter vector used in optimization routines back to the
        original PyTree structure used by the network. Otherwise,
        the network would not be able to use the parameters as it just 
        expects a PyTree structure (dict, list, custom).

        Parameters
        ----------
        flat_params : jnp.ndarray
            The flat parameter vector to transform.
        shapes : list
            The shapes of the original parameters.
        sizes : list
            The sizes of the original parameters.
        is_cpx : bool
            Whether the parameters are complex.

        Returns
        -------
        Any
            The transformed PyTree structure.
        """
        if not self._isjax:
            raise NotImplementedError("Only JAX backend supported.")
        # transform shapes to NamedTuple
        slices = net_utils.jaxpy.prepare_slice_info(shapes, sizes, is_cpx)
    
        # Transform the flat parameters back to the original PyTree structure
        return net_utils.jaxpy.transform_flat_params_jit(flat_params, self._params_tree_def, slices, self._params_total_size)
    
    def update_parameters(self, d_par: Any, mult: Any, shapes, sizes, iscpx):
        """
        Update model parameters using a flat vector (in real representation) or a PyTree.
        
        This function allows to provide a parameter change 'd_par' either as a flat
        JAX array or as a PyTree matching the model's parameter structure. The update
        is scaled by 'mult' before being applied to the current parameters.
        
        For example: 
            - when mult = -learning_rate, this function performs a gradient descent step.
            - when mult = +learning_rate, this function performs a gradient ascent step.
            - when mult = -i*dt, this function performs a real-time evolution step.
            - when mult = -dt, this function performs an imaginary-time evolution step.
        
        Parameters
        ----------
        d_par : Any
            The parameter update. Can be:
                1.  A 1D JAX array (`jnp.ndarray`) containing the update in the
                    flattened **real representation** format (matching the structure
                    defined by the model's parameters, including [Re, Im] for complex leaves).
                2.  A PyTree (dict, list, custom) matching the exact structure of
                    the model's parameters.
        """
        if not self._isjax:
            self._logger.warning("Only JAX backend supported for parameter updates.")
            return

        #! Handle Input Types
        if isinstance(d_par, jnp.ndarray):
            update_tree = self.transform_flat_params(d_par * mult, shapes, sizes, iscpx)    # Transform flat vector to PyTree structure
            
        elif isinstance(d_par, (dict, list)) or hasattr(d_par, '__jax_flatten__'):          # Validate PyTree structure (necessary for correctness)
            
            flat_leaves_dpar, tree_d_par            = tree_flatten(d_par * mult)
            if tree_d_par != self._params_tree_def: raise ValueError("Provided `d_par` PyTree structure does not match model parameters structure.")
            update_tree                             = d_par
        else:
            raise TypeError(f"Unsupported type for parameter update `d_par`: {type(d_par)}. Expected PyTree or 1D JAX array.")

        #! Update Parameters
        current_params  = self._net.get_params()
        new_params      = net_utils.jaxpy.add_tree(current_params, update_tree)
        self._net.set_params(new_params)
    
        # Aliases for update_parameters (similar to TensorFlow/PyTorch naming)
    
    def apply_gradients(self, d_par, mult, shapes, sizes, iscpx):
        """Alias for update_parameters, similar to TensorFlow's optimizer.apply_gradients."""
        return self.update_parameters(d_par, mult, shapes, sizes, iscpx)

    def step_optimizer(self, d_par, mult, shapes, sizes, iscpx):
        """Alias for update_parameters, similar to PyTorch's optimizer.step."""
        return self.update_parameters(d_par, mult, shapes, sizes, iscpx)
    
    #####################################
    #! TRAINING/EVO OVERRIDES - SINGLE STEP
    #####################################
    
    def _sample_for_shapes(self, *args, **kwargs):
        """
        Sample a configuration to get the shapes of the parameters.
        This is used to initialize the shapes of the parameters.
        """
        #! Sample the configurations
        numchains   = self._sampler.numchains
        numsamples  = self._sampler.numsamples
        params      = self.get_params()
        (_, _), (configs, configs_ansatze), probabilities = self._sampler.sample(params)
        
        #! Get the shapes of the parameters
        result = NQS._single_step_groundstate(params, configs, configs_ansatze, probabilities, *args, **kwargs)
        
        #! Set the number of chains and samples
        self._sampler.set_numchains(numchains)
        self._sampler.set_numsamples(numsamples)
        return result.params_shapes, result.params_sizes, result.params_cpx
    
    def wrap_single_step_jax(self, batch_size: Optional[int] = None):
        """
        Wraps the single-step JAX function for use in optimization or sampling routines.
        This method prepares and returns a JIT-compiled function that performs a single optimization or sampling step
        using the neural quantum state (NQS) ansatz and associated functions. It handles parameter transformation,
        function initialization, and batching.
        Args:
            batch_size (Optional[int]): The batch size to use for sampling configurations. If not provided,
                the default batch size (`self._batch_size`) is used.
        Returns:
            Callable: A JIT-compiled function with the signature:
                wrapped(y, t, configs, configs_ansatze, probabilities, int_step=0)
            where:
                - y: Flat parameter vector.
                - t: Current step or time (used as a static argument for JIT).
                - configs: Sampled configurations.
                - configs_ansatze: Ansatz-specific configurations.
                - probabilities: Probability weights for the configurations.
                - int_step: (Optional) Integer step counter (default: 0).
        Notes:
            - The returned function automatically transforms flat parameters into the required tree structure.
            - All necessary NQS functions are (re)initialized to ensure correct compilation.
            - This wrapper is intended for use in iterative algorithms such as VMC or optimization loops.
        """
        
        batch_size                  = batch_size if batch_size is not None else self._batch_size
        
        #! reinitialize the functions - it may happen that they recompile but that doesn't matter
        ansatz_fn                   = self._ansatz_func
        local_energy_fn             = self._local_en_func
        flat_grad_fn                = self._flat_grad_func
        apply_fn                    = self._apply_func
        compute_grad_f              = net_utils.jaxpy.compute_gradients_batched if not self._analytic else self._grad_func
        
        #! Sample the configurations
        # self._init_param_metadata()
        shapes, sizes, iscpx        = self._sample_for_shapes(ansatz_fn, apply_fn, local_energy_fn, flat_grad_fn, compute_grad_f, batch_size=self._batch_size, t=0)
        # shapes, sizes, iscpx = self._params_shapes, self._params_sizes, self._params_iscpx
        tree_def, flat_size, slices = self._params_tree_def, self._params_total_size, net_utils.jaxpy.prepare_slice_info(shapes, sizes, iscpx)

        #! Create the function to be used
        single_step_jax = partial(NQS._single_step_groundstate, ansatz_fn = ansatz_fn, local_energy_fn = local_energy_fn,
                                flat_grad_fn = flat_grad_fn, apply_fn = apply_fn, batch_size = batch_size, compute_grad_f = compute_grad_f)

        #! prepares the wrapped function
        @partial(jax.jit, static_argnames=('t',))
        def wrapped(y, t, configs, configs_ansatze, probabilities, int_step = 0):
            params = net_utils.jaxpy.transform_flat_params_jit(y, tree_def, slices, flat_size)
            result = single_step_jax(params, configs, configs_ansatze, probabilities, t=t, int_step=int_step)
            return (result.loss, result.loss_mean, result.loss_std), result.grad_flat, (shapes, sizes, iscpx)
        return wrapped

    @staticmethod
    def _single_step_groundstate(params         : Any, 
                                configs         : Array, 
                                configs_ansatze : Any, 
                                probabilities   : Any, 
                                # functions (Static input for JIT)
                                ansatz_fn       : Callable  = None,         # ansatz function to evaluate the log wave-function
                                apply_fn        : Callable  = None,         # apply function to evaluate the local energy
                                local_energy_fn : Callable  = None,         # local energy function
                                flat_grad_fn    : Callable  = None,         # function to compute the flattened gradient for one sample
                                compute_grad_f  : Callable  = net_utils.jaxpy.compute_gradients_batched, # function to compute the gradients in batch - jax or numpy
                                # Static for evaluation
                                batch_size      : int       = None,         # batch size for evaluation
                                t               : float     = None,         # time for the jax
                                int_step        : int       = 0) -> NQSSingleStepResult:
        '''
        Perform a single training step to obtain the energies and gradients of the NQS.
        This method computes the local energies, gradients, and other relevant metrics for a single step of
        training or evaluation of the Neural Quantum State (NQS) model. It can utilize either JAX or NumPy
        for computations, depending on the backend in use.

        Parameters:
            params: 
                The parameters (weights) to use for the network evaluation.
            configs: 
                The sampled configurations.
            configs_ansatze: 
                The ansatze associated with the sampled configurations.
            probabilities: 
                The probabilities associated with the sampled configurations.
            batch_size: 
                The size of batches to use for the evaluation.
            *args:
                Additional positional arguments.
            **kwargs:
                Additional keyword arguments.
        Returns:
            A tuple containing:
                - energies: The computed local energies:
                    - v: Local energies for each configuration.
                    - means: Mean of the local energies.
                    - stds: Standard deviation of the local energies.
                - gradients: The computed gradients of the log wave-function - flattened.
                - param_shapes: Shapes of the parameters:
                    - shapes: List of shapes of the parameters.
                    - sizes: List of sizes of the parameters.
                    - iscpx: Boolean indicating if the parameters are complex.
        '''

        #! a) perform the single step - calculate energies
        (v, means, stds) = apply_fn(func            = local_energy_fn,  # local_energy_fun,
                                    states          = configs,          # estimate on those configs
                                    sample_probas   = probabilities,    # associated probabilities
                                    logprobas_in    = configs_ansatze,  # associated ansatze - log(psi(s))
                                    logproba_fun    = ansatz_fn,        # log(psi(s)) function
                                    parameters      = params,           # network parameters
                                    batch_size      = batch_size) # batch size
        
        #! b) compute the gradients O_k = ∇ log \psi 
        # The output `flat_grads` will have the dtype determined by single_sample_flat_grad_fun
        # For complex NQS, this is typically complex. Shape: (batch_size, n_params_flat)
        flat_grads, shapes, sizes, iscpx = compute_grad_f(net_apply     = ansatz_fn,
                                            params                      = params,
                                            states                      = configs,
                                            single_sample_flat_grad_fun = flat_grad_fn,
                                            batch_size                  = batch_size)

        return NQSSingleStepResult(loss = v, loss_mean=means, loss_std=stds,
                    grad_flat = flat_grads, params_shapes=shapes, params_sizes=sizes, params_cpx=iscpx)

    def step(self, problem: str = 'ground', 
            configs: Array = None, 
            configs_ansatze: Any = None, 
            probabilities: Any = None, 
            params: Any = None, **kwargs) -> NQSSingleStepResult:
        '''
        Perform a single step of the specified problem type.
        This method determines the type of problem (ground state, excited state, or time evolution)
        and calls the appropriate single-step function to perform the computation.
        
        Parameters:
            problem: The type of problem to solve ('ground', 'excited', 'time').
            **kwargs: Additional keyword arguments to pass to the single-step function.
        Returns:
            The result of the single-step computation, which may include energies, gradients, and other metrics
            depending on the problem type.        
        '''
        
        # try to reset batch size
        self._set_batch_size(kwargs.get('batch_size', self._batch_size))
        
        # check if the parameters are provided
        params              = self.get_params() if params is None else params
        
        # prepare the functions if not provided
        ansatz_fn           = kwargs.get('ansatz_fn',           self._ansatz_func)
        apply_fn            = kwargs.get('apply_fn',            self._apply_func)
        local_energy_fun    = kwargs.get('local_energy_fun',    self._local_en_func)
        flat_grad_fun       = kwargs.get('flat_grad_fun',       self._flat_grad_func)
        compute_grad_fun    = net_utils.jaxpy.compute_gradients_batched if self._isjax else None
        
        # check if the configurations are provided
        if configs is None or configs_ansatze is None or probabilities is None:
            num_samples     = kwargs.get('num_samples', self._sampler.numsamples)
            num_chains      = kwargs.get('num_chains', self._sampler.numchains)
            (_, _), (configs, configs_ansatze), probabilities = self._sampler.sample(parameters=params, 
                                                            num_samples=num_samples, num_chains=num_chains)
        
        
        # call the appropriate single step function
        if isinstance(self._nqsproblem, WavefunctionPhysics):
            if problem == 'ground':
                return self._single_step_groundstate(params=params, 
                                                    configs=configs, 
                                                    configs_ansatze=configs_ansatze, 
                                                    probabilities=probabilities, 
                                                    ansatz_fn=ansatz_fn,
                                                    apply_fn=apply_fn,
                                                    local_energy_fun=local_energy_fun,
                                                    flat_grad_fun=flat_grad_fun,
                                                    compute_grad_f=compute_grad_fun,
                                                    batch_size=self._batch_size,
                                                    **kwargs)
            elif problem == 'excited':
                raise NotImplementedError("Excited state calculation not implemented yet.")
                # return self._single_step_excitedstate(**kwargs)
        elif isinstance(self._nqsproblem, DensityMatrixPhysics):
            raise NotImplementedError("Time evolution not implemented yet.")
            # return self._single_step_timeevolution(**kwargs)
        else:
            raise ValueError("Unknown problem type.")
    
    #####################################
    #! STATE MODIFIER
    #!TODO: THIS IS NOT FINISHED, MUST BE COMPLETED
    #####################################
    
    @property
    def modifier(self) -> Union[Operator, OperatorFunction]:
        '''
        Return the state modifier.
        '''
        return self._modifier
    
    @property
    def modified(self) -> bool:
        '''
        Return True if the state is modified, False otherwise.
        '''
        return self._modifier is not None
    
    @property
    def ansatz_modified(self):
        '''
        Return the ansatz function with the modifier applied.
        '''
        return self._ansatz_mod_func
    
    def unset_modifier(self):
        '''
        Unset the state modifier.
        '''
        self._modifier = None
        self.log("State modifier unset.", log='info', lvl = 2, color = 'blue')
        
        # reset the ansatz function
        self._ansatz_func, self._params = self._net.get_apply(self._isjax)
    
    def _set_modifier_func(self):
        '''
        Set the state modifier function.
        '''
        if self._modifier is None:
            self.log("State modifier is None. Cannot set the function.", log='error', lvl = 2, color = 'red')
            return

        if isinstance(self._modifier, Operator):
            if self._isjax and hasattr(self._modifier, 'jax'):
                self._modifier_func = self._modifier.jax
            elif hasattr(self._modifier, 'npy'):
                self._modifier_func = self._modifier.npy
            else:
                raise ValueError("The operator does not have a JAX or NumPy implementation.")
        else:
            self._modifier_func = self._modifier
        self.log(f"State modifier function set to {self._modifier}.", log='info', lvl = 2, color = 'blue')

    def set_modifier(self, modifier: Union[Operator, OperatorFunction], **kwargs):
        '''
        Set the state modifier.
        '''
        
        #! The modifier should be an inverse mapping - for a given state, it shall return all the states that lead to it 
        #! through the application of the operator.
        self._modifier = modifier
        self._set_modifier_func()
        
        #! get the ansatz function without the modifier
        model_callable, self._params = self._net.get_apply(self._isjax)
        
        # it modifies this ansatz function, one needs to recompile and take it
        def _ansatz_func_jax(params, x):
            """
            Args
            ----
            params : pytree - network parameters
            x      : (..., N_dim) - either a single state (N_dim,) or a batch (N_sample, N_dim)

            Returns
            -------
            ansatz : (N_sample,) or () - product of network x modifier for every sample
            """
            x           = jnp.atleast_2d(x)     # (B, N_dim);  B = 1 for a single state
            B           = x.shape[0]            
            
            #! obtain all modified states & weights
            #    st  : (B, M, N_dim)
            #    w   : (B, M)
            st, w       = jax.vmap(self._modifier_func)(x)
            M           = st.shape[1]                                               # number of modified states
            #! flatten (B * M) so we can call the network once
            st_flat     = st.reshape(-1, st.shape[-1])                              # (B\cdot M, N_dim)
            log_psi     = jax.vmap(lambda s: model_callable(params, s))(st_flat)    # (B\cdot M,)
            #! reshape the log_psi to (B, M)
            log_psi_r   = log_psi.reshape(B, -1)                                    # (B, M)
            log_psi_r  += jnp.log(w.astype(log_psi_r.dtype))

            #! exponentiate the log_psi to combine the weights
            def _many_mods(lpr): # lpr: (B, M)
                return jnp.log(jnp.sum(jnp.exp(lpr), axis=1))                       # (B,)
            
            def _one_mod(lpr):
                return lpr[:, 0]                                                    # (B,)
            
            ansatz = jax.lax.cond(
                M > 1,
                _many_mods,
                _one_mod,
                log_psi_r
            )
            return ansatz
        
        if self._isjax:
            self._ansatz_mod_func = jax.jit(_ansatz_func_jax)
        else:
            self.log("JAX backend is not available. Cannot set the ansatz function.", log='error', lvl = 2, color = 'red')
            self._ansatz_mod_func = model_callable
        
        #! set the ansatz function to the modified one
        self._ansatz_func = self._ansatz_mod_func
    
    #####################################
    #! UPDATES
    #####################################
    
    def update(self, **kwargs):
        '''
        Update the NQS solver after state modification.
        '''
    
    def unupdate(self, **kwargs):
        '''
        Unupdate the NQS solver after state modification.
        '''

    #####################################
    #! WEIGHTS
    #####################################
    
    def save_weights(
        self,
        filename      : Optional[Directories]   = None,
        fmt           : str                     = "h5",
        absolute      : bool                    = False,
        step          : Optional[int]           = 0,
        max_to_keep   : Optional[int]           = 1,
        overwrite     : bool                    = True,
        save_metadata : bool                    = False) -> Dict[str, Any]:
        """
        Save network weights to disk.

        Args:
            filename:
                Name of the file (with extension). If None, defaults to '<net>_weights.<fmt>'.
            fmt:
                Format ('h5' or 'npz').
            absolute:
                If True, treat filename as full path; if False, place under self._dir_detailed.
            use_orbax:
                If True and net is FlaxNetInterface, use Orbax CheckpointManager.
            overwrite:
                If True, overwrite existing file.
            save_metadata:
                If True, also write '<filename>.json' with repr(self) metadata.

        Returns:
            Full path to the saved weights file.
        """
        
        import os
        from pathlib import Path
        
        # determine filename
        if filename is None:
            base_name = f"{type(self._net).__name__}_weights"
            filename  = f"{base_name}.{fmt}"
        
        # Resolve Path
        if absolute:
            path = Path(filename)
        else:
            # Fallback to internal dir logic if needed
            base = Path(self._dir_detailed) # Assuming this attribute exists
            path = base / filename
            
        # ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not overwrite:
            print(f"Checkpoint {path} exists, skipping.")
            return

        # get current parameters
        params = self._net.get_params()

        if self._use_orbax and self._ckpt_manager:
            if step is None:
                step = 0
                
            # Optionally override keep parameter
            if max_to_keep is not None:
                self._ckpt_manager.max_to_keep = max_to_keep

            self._ckpt_manager.save(
                        step        = step,
                        items       = { 'params': params },
                        overwrite   = overwrite
                    )
            out_path = self._ckpt_manager.directory
            info     = {
                        'path'      : out_path,
                        'step'      : step,
                        'max_kept'  : self._ckpt_manager.max_to_keep
                    }
        else:
            import h5py
            
            if fmt == 'h5':
                with h5py.File(path, 'w') as f:
                    # Recursively write PyTree (Dict of Arrays)
                    def write_group(h5_group, py_dict):
                        for k, v in py_dict.items():
                            if isinstance(v, dict) or hasattr(v, 'items'):
                                sub_group = h5_group.create_group(k)
                                write_group(sub_group, v)
                            else:
                                h5_group.create_dataset(k, data=np.array(v))
                    
                    write_group(f, params)
                    
                    # Save Metadata inside H5 (Cleaner than separate JSON)
                    if save_metadata and isinstance(save_metadata, dict):
                        for k, v in save_metadata.items():
                            # HDF5 attributes must be primitive types
                            try:
                                f.attrs[k] = v
                            except TypeError:
                                f.attrs[k] = str(v)
            elif fmt == 'npz':
                flat = {k: np.array(v) for k, v in params.items()}
                np.savez(path, **flat)
            else:
                raise ValueError(f"Unsupported format: {fmt}")
            
    def load_weights(
        self,
        filename : Optional[str] = None,
        fmt      : str           = "h5",
        absolute : bool          = False,
        step     : Optional[int] = None,
        use_orbax: bool          = False) -> Dict[str, Any]:
        """
        Load weights at given step (or latest) and set on self._net.

        Returns the loaded parameters dict.
        """
        if use_orbax and self._ckpt_manager:
            if step is None:
                #! load latest
                ckpts   = self._ckpt_manager.list_checkpoints()
                if not ckpts:
                    raise FileNotFoundError("No checkpoints found.")
                step    = max(ckpts)

            state       = self._ckpt_manager.restore(step=step)
            params      = state['params']
        else:
            base_name   = filename or f"{type(self._net).__name__}_weights"
            fname       = f"{base_name}.{fmt}" if not base_name.endswith(f".{fmt}") else base_name
            if absolute:
                path    = Path(fname)
            else:
                path    = Path(str(Directories(self._dir_detailed).join(fname)))
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            if fmt == 'h5':
                with h5py.File(path, 'r') as f:
                    def read_tree(node):
                        out = {}
                        for k, v in node.items():
                            out[k] = read_tree(v) if isinstance(v, h5py.Group) else v[()]
                        return out
                    params = read_tree(f)
            elif fmt == 'npz':
                data   = np.load(path)
                params = {k: data[k] for k in data.files}
            else:
                raise ValueError(f"Unsupported format: {fmt}")

        self.set_params(params)
        return params

    #####################################
    #! GET/SET PARAMETERS
    #####################################
    
    def get_params(self, unravel: bool = False) -> Any:
        """Returns the current parameters from the network object."""
        params = self._net.get_params()
        if unravel:
            return jnp.concatenate([p.ravel() for p in tree_flatten(params)[0]])
        return params
    
    def set_params(self, new_params: Any, shapes: list = None, sizes: list = None, iscpx: bool = False):
        """
        Sets new parameters in the network object.
        Parameters
        ----------
        new_params : Any
            The new parameters to set. Can be:
                1.  A PyTree (dict, list, custom) matching the exact structure of
                    the model's parameters.
                2.  A 1D JAX array (`jnp.ndarray`) containing the update in the
                    flattened **real representation** format (matching the structure
                    defined by the model's parameters, including [Re, Im] for complex leaves).
        """
        params = new_params
        
        # check if the parameters are provided
        if params is None:
            params = self._net.get_params()
        elif isinstance(params, jnp.ndarray):
            shapes = shapes if shapes is not None else self._params_shapes
            sizes  = sizes if sizes is not None else self._params_sizes
            iscpx  = iscpx if iscpx is not None else self._params_is_cpx
            params = self.transform_flat_params(params, shapes, sizes, iscpx)

        # set the parameters
        self._net.set_params(params)
    
    #####################################
    #! GETTERS AND PROPERTIES
    #####################################
    
    @property
    def beta_penalty(self):
        '''
        Return the beta penalty.
        '''
        return self._beta_penalty
    
    @property
    def net(self):
        '''
        Return the neural network.
        '''
        return self._net
    
    @property
    def net_flax(self):
        '''
        Return the neural network in Flax format.
        '''
        try:
            return self._net.net_module
        except AttributeError:
            raise AttributeError("The neural network is not in Flax format.")
        except Exception as e:
            raise e
        return None
    
    @property
    def num_params(self):
        '''
        Return the number of parameters in the neural network.
        '''
        return self._params_total_size
    
    @property
    def nvisible(self):
        '''
        Return the number of visible units in the neural network.
        '''
        return self._nvisible
    
    @property
    def size(self):
        '''
        Return the size of the neural network.
        '''
        return self._size
    
    @property
    def batch_size(self):
        '''
        Return the batch size.
        '''
        return self._batch_size
    
    @property
    def backend(self):
        '''
        Return the backend used for the neural network.
        '''
        return self._backend
    
    @property
    def nqsbackend(self):
        '''
        Return the backend used for the neural network.
        '''
        return self._nqsbackend
    
    #! Aliases for local energy
    @property
    def local_energy(self):
        '''
        Return the local energy function.
        '''
        return self._local_en_func
    @property
    def loc_energy(self):
        ''' Alias for local_energy '''
        return self._local_en_func
    @property
    def local_en(self):
        ''' Alias for local_energy '''
        return self._local_en_func
    
    @property
    def nvis(self):
        '''
        Return the number of visible units in the neural network.
        '''
        return self._nvisible
    
    @property
    def npar(self):
        '''
        Return the number of parameters in the neural network.
        '''
        return self._params_total_size
    
    #####################################

    def clone(self):
        ''' Clone the NQS solver. '''
        return NQS(self._net.clone(), self._sampler.clone(), self._backend, **self._kwargs)
    
    def swap(self, other):
        ''' Swap the NQS solver with another one. '''
        return super().swap(other)
    
    #####################################
    
    def __repr__(self):
        return f"NQS(ansatz={self._net},sampler={self._sampler},backend={self._backend_str})"
    
    def __str__(self):
        return f"NQS(ansatz={self._net},sampler={self._sampler},backend={self._backend_str})"    

#########################################
#! EOF
#########################################