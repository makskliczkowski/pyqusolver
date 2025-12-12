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

import os
import time
import warnings

# Some Globals for safety
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]     = "false" 
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]    = "0.7"
warnings.filterwarnings("ignore",               message = ".*Skipped cross-host ArrayMetadata validation.*")

# typing and other imports
from typing         import Union, Tuple, Union, Callable, Optional, Any, List, Callable, TYPE_CHECKING
from functools      import partial
from dataclasses    import dataclass

# import physical problems
try:
    from .src.nqs_physics                       import *
    from .src.nqs_network_integration           import *
    from .src.nqs_ansatz_modifier               import AnsatzModifier
    from .src.nqs_checkpoint_manager            import NQSCheckpointManager
    from .src.nqs_engine                        import NQSEvalEngine, NQSObservable, NQSLoss
except ImportError as e:
    raise ImportError("Failed to import nqs_physics or nqs_networks module. Ensure QES.NQS is correctly installed.") from e

# ----------------------------------------------------------

if TYPE_CHECKING:
    from QES.Algebra.Operator.operator_loader   import OperatorModule
    from QES.NQS.src.nqs_train                  import NQSTrainer, NQSTrainStats
    from QES.NQS.src.nqs_train                  import NQSTrainer, NQSTrainStats

# from QES.general_python imports
try:
    #! Algebra
    from QES.Algebra.Operator.operator          import Operator
    
    #! Randomness
    from QES.general_python.common.directories  import Directories
    from QES.general_python.common.flog         import Logger
    
    #! Monte Carlo
    from QES.Solver.MonteCarlo.sampler          import Sampler, get_sampler
    from QES.Solver.MonteCarlo.vmc              import VMCSampler
    from QES.Solver.MonteCarlo.montecarlo       import MonteCarloSolver
    
    #! Hilbert space
    from QES.Algebra.hilbert                    import HilbertSpace
    
    #! Choose network
    from QES.general_python.ml.networks         import choose_network as nqs_choose_network, GeneralNet
    
except ImportError as e:
    warnings.warn("Some QES.general_python modules could not be imported. Ensure QES.general_python is installed correctly.", ImportWarning)
    raise e

# ----------------------------------------------------------
AnsatzFunctionType      = Callable[[Callable, Array, int, Any], Array]                  # func, states, batch_size, params -> Array [log ansatz values]
ApplyFunctionType       = Callable[[Callable, Array, Array, Array, int, Any], Array]    # func, states, probs, params, batch_size -> Array [sampled/evaluated values]
EvalFunctionType        = Callable[[Callable, Array, Any, int], Array]                  # func, states, params, batch_size -> Array [log ansatz values]
CallableFunctionType    = Callable[[Array], Tuple[Array, Array]]                        # states -> (new_states, coeffs)
LossFunctionType        = CallableFunctionType                                          # states -> (new_states, coeffs)

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
    
    Parallel Tempering (PT)
    -----------------------
    When `replica > 1` is passed, the sampler automatically enables Parallel Tempering:
    
        - Multiple replicas run MCMC at different temperatures (betas).
        - A geometric temperature ladder is generated: `betas = logspace(0, log10(0.1), replica)`.
        - Periodic replica exchanges between adjacent temperatures improve mixing.
        - Only samples from the physical replica (beta=1.0) are used for training.
    
    This is useful for systems with rugged energy landscapes where standard MCMC
    gets trapped in local minima. PT allows sampling at higher temperatures where
    barriers are easier to cross, then exchanges bring low-energy configurations
    to the physical temperature.
    
    Example (PT mode):
    >>> nqs = NQS(logansatz=net, model=hamiltonian, shape=(4,4), replica=5)
    >>> nqs.train(...)  # Uses 5 temperature replicas with geometric ladder

    Examples (Physics & Sampling):
    >>> # Spin-1/2 (Local Updates)
    >>> # Standard VMC with single spin flip updates
    >>> model       = Hamiltonian(...)
    >>> nqs         = NQS(logansatz=RBM, model=model, upd_fun="local")

    >>> # Spinless Fermions (Particle Conservation) -> Exchange Updates
    >>> # Requires 'hilbert' to define the particle number sector
    >>> hilbert     = HilbertSpace(lattice=..., local_space=...)
    >>> model       = Hamiltonian(...)
    >>> # Exchange updates preserve particle number from initial state
    >>> nqs         = NQS(logansatz=..., model=model, hilbert=hilbert, 
    >>>             upd_fun="exchange", initstate="RND_FIXED", n_particles=5)

    >>> # Frustrated Systems (Global Updates)
    >>> # Use global updates to escape local minima
    >>> patterns    = model.lattice.calculate_plaquettes()
    >>> nqs         = NQS(logansatz=..., model=model, upd_fun="global", patterns=patterns, 
    >>>             p_global=0.5) # 50% global flips, 50% local flips

    Initialization & Updates
    ------------------------
    **Initial States (`initstate`):**
    - "RND"         : Random configuration (default).
    - "F_UP"        : All spins Up (+1).
    - "F_DN"        : All spins Down (-1/0).
    - "AF"          : Antiferromagnetic (Neel) state.
    - "RND_FIXED"   : Random state with fixed magnetization/particle number (requires `n_particles` or `magnetization`).

    **Update Rules (`upd_fun`):**
    - "LOCAL"       : Single spin flip (standard Metropolis).
    - "EXCHANGE"    : Swaps two sites. Preserves symmetries (U(1), Z_tot). 
                    Supports `exchange_order=k` for k-th neighbor exchange.
    - "BOND_FLIP"   : Flips a site and one random neighbor. Useful for moving topological excitations.
    - "GLOBAL"      : Flips a predefined pattern (e.g., plaquette). Requires `patterns` argument.
    - "MULTI_FLIP"  : Randomly flips `s_n_flip` sites at once.
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
    _ERROR_SHAPE_HILBERT        = "Either shape or hilbert space must be provided."
    _ERROR_ENERGY_WAVEFUNCTION  = "Energy computation is only valid for wavefunction problems."
    
    @staticmethod
    def DUMMY_APPLY_FUN_NPY(x): return np.array([np.array([x])]), np.array([np.array([1.0])])
    @staticmethod
    def DUMMY_APPLY_FUN_JAX(x): return jnp.array([jnp.array([x])] * 40), jnp.array([jnp.array([1.0])] * 40)
    
    def __init__(self,
                # information on the NQS
                logansatz   : Union[Callable, str, GeneralNet],
                model       : Hamiltonian,
                # information on the Monte Carlo solver
                sampler     : Optional[Union[Callable, str, VMCSampler]]    =   None,
                batch_size  : Optional[int]                                 =   1,
                nthstate    : Optional[int]                                 =   0,
                *,
                # information on the NQS
                nparticles  : Optional[int]                                 = None,
                # information on the Monte Carlo solver     
                seed        : Optional[int]                                 = None,
                beta        : float                                         = 1,
                mu          : float                                         = 2.0,
                replica     : int                                           = 1,
                # information on the NQS - Hilbert space
                shape       : Optional[Union[list, tuple]]                  = None,
                hilbert     : Optional[Union[HilbertSpace, list, tuple]]    = None,
                modes       : int                                           = 2,
                # information on the Monte Carlo solver
                directory   : Optional[str]                                 = MonteCarloSolver.defdir,
                backend     : str                                           = 'default',
                dtype       : Optional[Union[type, str]]                    = None,
                problem     : Optional[Union[str, PhysicsInterface]]        = 'wavefunction',
                # logging
                logger      : Optional[Logger]                              =   None,
                **kwargs):
        '''
        Initialize the NQS solver.
        
        Parameters:
            logansatz [Union[Callable, str, GeneralNet]]:
                The neural network or callable to be used. This can be specified in several ways:
                - As a string (e.g., 'rbm', 'cnn') to use a built-in network.
                - As a pre-initialized network object (must be a subclass of `GeneralNet`).
                - As a raw Flax module class for custom networks. The factory will wrap it automatically.
                See the documentation of `QES.general_python.ml.networks.choose_network` for details
                on custom module requirements.
                - As a callable that returns an ansatz. For example, a function that takes input shape and returns
                    a network instance or it can be some custom logic to create the wavefunction ansatz:
                    - PEPS ansatz function
                    - Custom variational ansatz
            sampler [Union[Callable, str, VMCSampler]]:
                The sampler to be used. If None, a default `VMCSampler` will be created.
                This can be specified in several ways:
                - As a string (e.g., 'vmc') to use a built-in sampler.
                - As a pre-initialized sampler object (must be a subclass of `VMCSampler`).
                - As a callable that returns a sampler instance.
                - If None, a default `VMCSampler` will be created using the provided network:
                    - The sampler will be initialized with the network, shape, random number generators, and other parameters.
            model [Hamiltonian or Operator]:
                The physical model (e.g., Hamiltonian) for the NQS.
                - If physics is 'wavefunction', this must provide a `local_energy_fn`.
                - For other physics types, refer to the specific requirements.
            batch_size [Optional[int]]:
                The batch size for training.
            nthstate [Optional[int]]:
                The nth excited state to target (0 for ground state). Informative
            nparticles [Optional[int]]:
                The number of particles in the system. If not provided, defaults to system size.
            seed [Optional[int]]:
                Random seed for initialization.
            beta [float]:
                Inverse temperature for Monte Carlo sampling.
            mu [float]:
                Mu parameter for the NQS. This is used for collecting different sampling distribution.
                By default, mu=2.0 corresponds to sampling from |psi(s)|^2 -> Born rule.
            replica [int]:
                Number of replicas for Parallel Tempering (PT). Default is 1 (PT disabled).
                When replica > 1:
                - A geometric temperature ladder is automatically generated from beta=1.0 (physical)
                to beta=0.1 (hot), unless custom `pt_betas` is provided via kwargs.
                - MCMC runs in parallel across all temperature replicas.
                - Periodic replica exchanges improve mixing and help escape local minima.
                - Only samples from the physical replica (beta=1.0) are used for training.
                - The return signature of `sample()` and `train()` remains unchanged.
            shape [Union[list, tuple]]:
                Shape of the input data, e.g., `(n_spins,)`.
            modes:
                Number of local modes per site (e.g., 2 for spin-1/2).
            directory:
                Directory for saving results.
            backend:
                Computational backend ('jax' or 'numpy').
            dtype:
                Data type for network parameters (e.g., 'float32', 'complex128').
            problem:
                The physics problem to solve (e.g., 'wavefunction').
            **kwargs:
                Additional keyword arguments passed to the network constructor.
                Including:
                - seed: 
                    Seed for network initialization (overrides global seed if provided).
                - param_dtype:
                    Data type for network parameters (e.g., 'float32', 'complex128').
                - beta_penalty: 
                    Penalty term for particle number conservation (default: 0.0).
                - use_orbax:
                    Whether to use Orbax for checkpointing (default: True).
                - orbax_max_to_keep:
                    Maximum number of Orbax checkpoints to keep (default: 3).
                - Sampler-specific parameters:
                    - s_numchains       : Number of Markov chains (default: 16).
                    - s_numsamples      : Number of samples per chain (default: 1000).
                    - s_numupd          : Number of updates per sample (default: 1). How many times a single update function is applied before collecting a sample.
                    - s_sweep_steps     : Number of sweep steps between samples (default: 10).
                    - s_therm_steps     : Number of thermalization steps (burnin) (default: 100).
                    - s_upd_fun         : Update function/rule for sampler (default: "LOCAL").
                                        Can be a string/Enum (e.g., "EXCHANGE", "GLOBAL") or a callable.
                                        See `NQS.help('sampling')` for details.
                    - s_patterns        : List of patterns for "GLOBAL" update rule (required if upd_fun="GLOBAL").
                    - s_n_flip          : Number of sites for "MULTI_FLIP" rule (default: 1).
                    - s_statetype       : Data type for sampler states (default: jnp.float32 or np.float32).
                    - s_logprob_fact    : Log probability factor (default: 0.5).
                    - s_makediffer      : Whether to make the sampler differentiable (default: True).
                    - Other sampler-specific parameters as needed.
                - Parallel Tempering parameters (when replica > 1):
                    - pt_betas          : Custom array of inverse temperatures (optional).
                                        If not provided, a geometric ladder from 1.0 to 0.1 is generated.
                    - pt_min_beta       : Minimum beta for auto-generated ladder (default: 0.1).
                                        Sampler-specific parameters (e.g., `global_p`, `global_fraction`) are
                                        also passed through to the `VMCSampler` constructor via `**kwargs`.
                    - global_p          : Probability of proposing a global update (default: 0.0).
                    - global_update     : Custom global update function (optional).
        '''
        
        if shape is None and hilbert is None:
            raise ValueError(self._ERROR_SHAPE_HILBERT)
        
        super().__init__(sampler    =   sampler,
                        seed        =   seed,
                        beta        =   beta,
                        mu          =   mu,
                        replica     =   replica,
                        shape       =   shape or ((model.lattice.ns,) if hasattr(model, 'lattice') else None),
                        hilbert     =   model.hilbert if hasattr(model, 'hilbert') else None,
                        modes       =   modes,
                        directory   =   directory,
                        backend     =   backend,
                        logger      =   logger,
                        dtype       =   dtype,
                        **kwargs)
                
        # --------------------------------------------------
        self._batch_size            = batch_size        
        self._initialized           = False
        self._seed                  = seed
        self._nthstate              = nthstate
        
        self._modifier_wrapper      = None   # type: Optional[AnsatzModifier]
        self._modifier_source       = None   # type: Optional[Union[Operator, Callable]]

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
        
        try:
            if logansatz is None:
                raise ValueError(self._ERROR_INVALID_NETWORK)
            
            self._logger.info(f"Initializing NQS network with ansatz: {logansatz}", lvl=1, color='blue')
            self._net = nqs_choose_network(logansatz, 
                        input_shape =   self._shape, 
                        backend     =   self._backend_str, 
                        dtype       =   dtype,
                        param_dtype =   kwargs.get('param_dtype', None),
                        seed        =   seed,
                        **kwargs
                    )
            
        except Exception as e:
            raise ValueError(f"Failed to initialize network. Check the network type and parameters.\nOriginal error: {e}")
        
        if not self._initialized:
            self.init_network()
        
        self._isjax                 = getattr(self._net, 'is_jax', (self._nqsbackend.name == "jax"))
        self._iscpx                 = self._net.is_complex
        self._holomorphic           = self._net.is_holomorphic
        self._analytic              = self._net.has_analytic_grad
        self._dtype                 = self._net.dtype
        
        # --------------------------------------------------
        #! Sampler
        # --------------------------------------------------
        
        self._model                 = model
        self._sampler               = self.set_sampler(sampler, kwargs.get("upd_fun", None), replica=replica, **kwargs)     

        # --------------------------------------------------
        #! Handle gradients
        # --------------------------------------------------
        self._grad_info             = self._nqsbackend.prepare_gradients(self._net)    
        self._flat_grad_func        = self._grad_info["flat_grad_func"]         # Function to compute flattened gradients
        self._analytic_grad_func    = self._grad_info["analytic_grad_func"]     # Function to compute analytic gradients
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
        self._ansatz_func, self._eval_func, self._apply_func    = self.nqsbackend.compile_functions(self._net, batch_size=self._batch_size)
        self._ansatz_base_func                                  = self._ansatz_func # Keep a reference to the pure ansatz function
        
        # --------------------------------------------------
        #! Model and physics setup
        # --------------------------------------------------

        self._nqsproblem.setup(self._model, self._net)
        # For wavefunction problem we keep the same attribute name you used:
        if self._nqsproblem.typ == 'wavefunction':
            self._local_en_func = getattr(self._nqsproblem, "local_energy_fn", None)
            self._loss_func     = self._local_en_func
        else:
            raise ValueError(self._ERROR_INVALID_PHYSICS)

        # --------------------------------------------------
        #! Initialize unified evaluation engine
        # --------------------------------------------------
        self._eval_engine = NQSEvalEngine(self, batch_size=batch_size, **kwargs)

        #######################################
        #! Directory to save the results
        #######################################
        
        self._init_directory()
        
        # Initialize the Manager
        self.ckpt_manager = NQSCheckpointManager(directory=self._dir_detailed, use_orbax=kwargs.get('use_orbax', True), max_to_keep=kwargs.get('orbax_max_to_keep', 3), logger=self._logger)
        
        # --------------------------------------------------
        #! Exact information (Ground Truth)
        # --------------------------------------------------
        self._exact_info            = None
    
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
        
        Directory Structure:
        --------------------
        {base_dir}/{model}/{lattice_or_shape}/{network_summary}/
        
        Note: Seed is NOT included in directory name (stored in metadata instead).
        This makes it easier to find and compare results across different seeds.
        """
        
        base                = Directories(self._dir)
        detailed            = base.join(str(self._model), create=False)

        #! add the lattice information if needed
        if self._model.lattice is not None:
            detailed        = detailed.join(str(self._model.lattice), create=False)
        else:
            detailed        = detailed.join(str(self.shape), create=False)

        #! network summary: skip sampler & params
        #   e.g. "RBM_shape=12_dtype=complex128"
        #   Note: seed is NOT included here (saved in metadata/stats instead)
        
        net_cls             = str(self._net)
        net_folder          = net_cls
        final_dir           = detailed.join(net_folder, create=False)

        #! actually mkdir them all
        #    create intermediate parents automatically
        base.mkdir()
        final_dir.mkdir()

        #! store for later use
        self._dir           = base
        self._dir_detailed  = final_dir
        self.defdir         = self._dir_detailed
        self.defdirpar      = self._dir_detailed.parent().resolve()
    
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
    
    @property
    def exact(self):
        """
        Return the exact information (ground truth) if available.
        This is a dictionary containing exact energy, method, etc.
        """
        return self._exact_info
    
    @exact.setter
    def exact(self, info: dict):
        """
        Set the exact information (ground truth).
        """
        if isinstance(info, dict):
            self._exact_info = info
            if info is not None:
                self.log(f"Exact information set: {info.get('exact_predictions', [None] * (self._nthstate + 1))[self._nthstate]}", lvl=1)
        elif isinstance(info, np.ndarray):
            self._exact_info = {
                'exact_predictions' : info,
                'exact_method'      : 'provided_array',
                'exact_energy'      : float(info) if np.ndim(info) == 0 else info[self._nthstate]
            }
            self.log(f"Exact information set from array: {self._exact_info['exact_energy']}", lvl=1)
        else:
            raise ValueError("Exact information must be provided as a dictionary or numpy array.")
        

    def get_exact(self, **kwargs) -> Optional['NQSTrainStats']:
        '''
        Get exact predictions, e.g., ground state energy, from the model.
        This method computes the exact ground state energy using the model's diagonalization method.
        It updates the trainer's statistics with the exact values if available.
        '''
        
        if self._nqsproblem.typ == 'wavefunction':
            stats = self.trainer.stats if self.trainer is not None else None
            
            if not stats.has_exact:
                if self.model.eig_val is None:
                    self.model.diagonalize(method='lanczos', k=kwargs.get('k', 10), store_basis=False, 
                                            verbose=kwargs.get('verbose', True), use_scipy=kwargs.get('use_scipy', True), 
                                            tol=kwargs.get('tol', 1e-7), max_iter=kwargs.get('max_iter', 200))
                pred                        =   self.model.eigenvalues
                if stats is not None:
                    stats.exact_predictions =   pred
                    
                nstate                      =   self._nthstate
                self.exact                  =   { 
                                                    'exact_predictions' : self.model.eigenvalues,
                                                    'exact_method'      : 'scipy_lanczos',
                                                    'exact_energy'      : float(pred) if np.ndim(pred) == 0 else pred[nstate]
                                            }
                if nstate == 0:
                    self._logger.info(f"Exact ground state energy: {self.model.eig_val[0]:.6f}", lvl=1, color='green')
                else:
                    self._logger.info(f"Exact state[{nstate}] energy: {self.model.eig_val[nstate]:.6f}", lvl=1, color='green')
                self._logger.info(f"Lowest energies: {self.model.eig_val[:max(nstate, 5)]}", lvl=2, color='green')
            return stats
        else:
            raise NotImplementedError("Exact is not implemented for other physics types yet...")               
        
    def load_exact(self, filepath: str, *, key: str = 'energy_values'):
        '''
        Load exact information from a file.
        This method loads the exact information (ground truth) from a specified file.
        Parameters:
            filepath: The path to the file containing the exact information.
            extension: The file extension (default is '.h5').
        '''
        
        if not os.path.isfile(filepath):
            self._logger.warning(f"Exact file {filepath} does not exist.", lvl=1, color='red')
            return
        
        # determine file extension
        extension = os.path.splitext(str(filepath))[1].lower()
        
        if extension == '.h5' or extension == '.hdf5':
            import h5py
            
            exact_values = []
            with h5py.File(filepath, 'r') as f:
                if key in f:
                    exact_values = f[key][:]
                else:
                    raise KeyError(f"Key '{key}' not found in HDF5 file '{filepath}'.")
                
        elif extension == '.npy':
            exact_values = np.load(filepath)
            
        elif extension == '.txt' or extension == '.dat' or extension == '.csv':
            exact_values = np.loadtxt(filepath)
        
        self.exact          = {
                                'exact_predictions' : np.array(exact_values),
                                'exact_method'      : f"loaded_from_{os.path.basename(filepath)}",
                                'exact_energy'      : float(exact_values) if np.ndim(exact_values) == 0 else exact_values[self._nthstate]
                            }
        self._logger.info(f"Loaded exact information from {filepath}.", lvl=1, color='green')
        
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
        self._batch_size                                        = batch_size
        self._ansatz_func, self._eval_func, self._apply_func    = self.nqsbackend.compile_functions(self._net, batch_size=self._batch_size)
        self._ansatz_base_func                                  = self._ansatz_func
    
    def set_sampler(self, sampler: Union[VMCSampler, str], upd_fun: Optional[Callable] = None, replica: int = 1, **kwargs) -> VMCSampler:
        ''' Set a new sampler for the NQS solver. '''  
        
        if isinstance(sampler, Sampler):
            self._sampler = sampler
            return self._sampler
        
        if sampler is None or isinstance(sampler, str):
    
            # DEFAULT FALLBACK
            if sampler is None: 
                sampler_type = 'MCSampler' # Default to MCMC
            else:
                sampler_type = sampler
    
            if self._isjax:
                import jax.numpy as jnp
                
            sampler_kwargs                  = kwargs.copy()
            sampler_kwargs['net']           = self._net
            sampler_kwargs['shape']         = self._shape
            sampler_kwargs['rng']           = self._rng
            sampler_kwargs['rng_k']         = self._rng_k
            sampler_kwargs['dtype']         = self._dtype            
            sampler_kwargs['beta']          = self._beta
            sampler_kwargs['mu']            = self._mu
            sampler_kwargs['hilbert']       = self._hilbert
            sampler_kwargs['backend']       = self._backend_str
            
            sampler_kwargs['numchains']     = kwargs.get('s_numchains', 16)
            sampler_kwargs.pop('s_numchains', None)
            sampler_kwargs['numsamples']    = kwargs.get('s_numsamples', 1000)
            sampler_kwargs.pop('s_numsamples', None)
            
            sampler_kwargs['numupd']        = kwargs.get('s_numupd', 1)
            sampler_kwargs.pop('s_numupd', None)
            
            sampler_kwargs['sweep_steps']   = kwargs.get('s_sweep_steps', 10)
            sampler_kwargs.pop('s_sweep_steps', None)
            
            sampler_kwargs['therm_steps']   = kwargs.get('s_therm_steps', 100)
            sampler_kwargs.pop('s_therm_steps', None)
            
            sampler_kwargs['statetype']     = kwargs.get('s_statetype', jnp.float32 if self._isjax else np.float32)
            sampler_kwargs.pop('s_statetype', None)
            
            sampler_kwargs['initstate']     = kwargs.get('s_initstate', None)
            sampler_kwargs.pop('s_initstate', None)
            
            sampler_kwargs['logprob_fact']  = kwargs.get('s_logprob_fact', 0.5)
            sampler_kwargs.pop('s_logprob_fact', None)
            
            sampler_kwargs['makediffer']    = kwargs.get('s_makediffer', True)
            sampler_kwargs.pop('s_makediffer', None)
            
            upd_fun                         = upd_fun or kwargs.get('s_upd_fun', upd_fun)
            sampler_kwargs.pop('s_upd_fun', None)
            
            # Pass logger to sampler
            sampler_kwargs['logger']        = self._logger
            sampler_kwargs['hilbert']       = self._hilbert
            sampler_kwargs['lattice']       = self._hilbert.lattice if self._model is not None else None
            
            sampler_kwargs['patterns']      = kwargs.get('s_patterns', None)
            sampler_kwargs.pop('s_patterns', None)
            sampler_kwargs['n_flip']        = kwargs.get('s_n_flip', 1)
            sampler_kwargs.pop('s_n_flip', None)
            
            # Global updates
            if 's_p_global' in kwargs: 
                sampler_kwargs['global_p'] = kwargs.pop('s_p_global')
            else:
                sampler_kwargs['global_p'] = kwargs.pop('s_global_p', 0.0)
                
            sampler_kwargs['global_update'] = kwargs.pop('s_global_update', None)
            
            sampler_kwargs['beta_penalty']  = self._beta_penalty
            sampler_kwargs.pop('s_beta_penalty', None)
            sampler_kwargs['n_particles']   = self._nparticles
            sampler_kwargs.pop('s_n_particles', None)

            # -----------------------------------------------------------------
            #! Parallel Tempering Setup
            # -----------------------------------------------------------------
            # If replicas > 1 is requested but no explicit pt_betas provided,
            # automatically generate a geometric temperature ladder
            n_replicas  = replica
            pt_betas    = kwargs.get('pt_betas', None)
            if n_replicas > 1 or pt_betas is not None:
                sampler_kwargs['pt_replicas']   = n_replicas
                sampler_kwargs['pt_betas']      = pt_betas
                self._logger.info(f"Parallel Tempering enabled with {n_replicas if pt_betas is None else pt_betas} replicas.", lvl=1, color='blue')
                
            # -----------------------------------------------------------------

            self._sampler = get_sampler(sampler_type, upd_fun=upd_fun, **sampler_kwargs)
            self._logger.warning(f"Using {self._sampler}", lvl=0, color='blue')
        else:
            self._sampler = get_sampler(sampler, self._net, self._shape, self._rng, self._rng_k, dtype=self._dtype, upd_fun=upd_fun, **kwargs)
        
        return self._sampler        

    #####################################
    #! EVALUATION OF THE ANSATZ BATCHED (\psi(s))
    #####################################
    
    def evaluate(self, states, *, batch_size=None, params=None, num_samples=None, num_chains=None, return_stats=False, return_values=False, **kwargs):
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
        states = self._nqsbackend.asarray(states, dtype = self._sampler._statetype)
        result = self._eval_engine.ansatz(states=states, params=params, batch_size=batch_size, num_samples=num_samples, num_chains=num_chains, return_stats=return_stats, return_values=return_values, **kwargs)
        return result if return_stats else result.values
    
    def ansatz(self, states, *, batch_size=None, params=None, num_samples=None, num_chains=None, return_stats=False, return_values=False, **kwargs):
        ''' Alias for the log ansatz evaluation '''
        return self.evaluate(states, batch_size=batch_size, params=params, num_samples=num_samples, num_chains=num_chains, return_stats=return_stats, return_values=return_values, **kwargs)
    
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
        states = self._nqsbackend.asarray(states, dtype = self._sampler._statetype)
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
    @partial(jax.jit, static_argnames=['func', 'logproba_fun', 'batch_size'])
    def _apply_fun_jax(func   : Callable,                           # function to be evaluated (e.g., local energy f: s -> E_loc(s))
                states        : Array,                              # input states (shape: (N, ...))
                probabilities : Array,                              # probabilities associated with the states (shape: (N,))
                logproba_in   : Array,                              # logarithm of the probabilities for the input states (\log p(s))
                logproba_fun  : Callable,                           # function to compute the logarithm of probabilities (\log p(s') -> to evaluate)
                parameters    : Union[dict, list, jnp.ndarray],     # parameters to be passed to the function - for the Networks ansatze
                batch_size    : Optional[int] = None,               # batch size for evaluation
                *op_args):                                          # additional arguments to pass to func (broadcast across states)
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
            *op_args:
                Additional arguments to pass to func. These are broadcast (not vmapped) across states.
                Useful for operator indices (i, j) in correlation functions.
        Returns:
            The result of the function evaluation, either batched or unbatched, depending on the value of `batch_size`.
        """
        if batch_size is None or batch_size == 1:
            funct_in = net_utils.jaxpy.apply_callable_jax
            return funct_in(func, states, probabilities, logproba_in, logproba_fun, parameters, 1, *op_args)
        else:
            funct_in = net_utils.jaxpy.apply_callable_batched_jax
            return funct_in(func, states, probabilities, logproba_in, logproba_fun, parameters, batch_size, *op_args)
    
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
            is_jax          : bool = True,
            *op_args):
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
            *op_args:
                Additional arguments to pass to func. These are broadcast (not vmapped) across states.
                Useful for operator indices (i, j) in correlation functions.
        Returns:
            The result of the function evaluation, either batched or unbatched, depending on the value of `batch_size`.
        """
                    
        if batch_size is None:
            funct_in = net_utils.jaxpy.apply_callable_jax if is_jax else net_utils.numpy.apply_callable_np
            return funct_in(func, states, probabilities, logproba_in, logproba_fun, parameters, 1, *op_args)
        
        # otherwise, we shall use the batched version
        funct_in = net_utils.jaxpy.apply_callable_batched_jax if is_jax else net_utils.numpy.apply_callable_batched_np
        return funct_in(func, states, probabilities, logproba_in, logproba_fun, parameters, batch_size, *op_args)
    
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
            functions       : Optional[Union[List, Callable]],
            *,
            states_and_psi  : Optional[Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray]]]   = None,
            probabilities   : Optional[Union[np.ndarray, jnp.ndarray]]                                          = None,
            batch_size      : Optional[int]                                                                     = None,
            parameters      : Optional[dict]                                                                    = None,
            num_samples     : Optional[int]                                                                     = None,
            num_chains      : Optional[int]                                                                     = None,
            return_values   : bool                                                                              = False,
            log_progress    : bool                                                                              = False,
            args            : Optional[tuple]                                                                   = None
            ):
        r"""
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
            log_progress (bool):
                Whether to log progress during the evaluation. Defaults to False.
            return_values (bool):
                If True, only the output of the evaluated functions is returned. If False, 
                returns a tuple containing the states, wavefunction, probabilities, and output. Defaults to False.
            parameters (Optional[dict]):
                The parameters (weights) to use for the network evaluation. If not provided, 
                defaults to the current parameters of the network (`self.get_params()`).
            batch_size (Optional[int]):
                The batch size to use for evaluation. If not provided, defaults to `self._batch_size`.
            num_samples (Optional[int]):
                Number of samples to generate if using the sampler. If not provided, 
                defaults to the sampler's `numsamples`.
            num_chains (Optional[int]):
                Number of chains to use if using the sampler. If not provided, 
                defaults to the sampler's `numchains`.
            args (Optional[tuple]):
                Additional arguments to pass to the function(s). These are broadcast (not vmapped) 
                across all sampled states. Useful for passing operator indices (i, j) in correlation 
                functions like <Sz_i Sz_j> without triggering recompilation for each (i, j) pair.
                Only 3 compilations needed: 0-arg, 1-arg, 2-arg versions, then reuse for all N² pairs.
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
                - Returns (states, ansatze), probabilities, output
                    - states: The sampled states from the sampler.
                    - ansatze: The corresponding wavefunction values for the sampled states.
                    - probabilities: The probabilities associated with the sampled states.
                    - The output is generally a list of: estimates, mean(estimates), std(estimates)
        Example:
            >>> (states, ansatze), probabilities, output = psi.apply([f1, ...], ...)
            where `f1` is a function that takes the required inputs and returns the desired output.
            - f1 (x) = [{|s'>}], [<s'|f|s> for s' in {|s'>}]
            and output is the O_loc(s) evaluated over the sampled states.
            - O_loc(s) = \sum _{s \in {samples}} sum_{s'} <s'|O|s> * psi(s') / psi(s) / num_samples
            
            # For correlation functions with operator indices:
            >>> def measure_corr(state, i, j):
            ...     # compute <Sz_i Sz_j> on state
            ...     return new_states, values
            >>> result = psi.apply(measure_corr, args=(jnp.array(0), jnp.array(1)))
        """
        
        params          = parameters    if parameters   is not None else self.get_params()
        batch_size      = batch_size    if batch_size   is not None else self._batch_size
        op_args         = args          if args         is not None else ()
        
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
                if states is None:
                    states_and_psi = None           # fallback to sampler
                else:
                    ansatze = self(states) if ansatze is None else ansatze
            elif isinstance(states_and_psi, np.ndarray) or isinstance(states_and_psi, jnp.ndarray):
                states          = states_and_psi    # assume it's the states
                ansatze         = self(states)      # call log ansatz function
            else:
                raise ValueError(self._ERROR_STATES_PSI)
        
        if states_and_psi is None:
            # get other parameters from kwargs
            num_samples                         = num_samples or self._sampler.numsamples
            num_chains                          = num_chains  or self._sampler.numchains
            _, (states, ansatze), probabilities = self._sampler.sample(parameters=params, num_samples=num_samples, num_chains=num_chains)
            
        # check if the probabilities are provided
        if probabilities is None:
            probabilities = self._backend.ones_like(ansatze).astype(ansatze.dtype)
            
        output = []
        for i, f in enumerate(functions):
            
            if log_progress:
                t0 = time.time()
            
            result = self._apply_func(
                f,
                states,
                probabilities,
                ansatze,
                self._ansatz_func,
                params,
                batch_size,
                *op_args
            )
            
            if log_progress:
                self._logger.info(f"Function ({i+1}/{len(functions)}) applied in {(time.time() - t0)*1000:.2f} ms", lvl=3, color='green')
                
            output.append(result)
        
        # check if the output is a list
        if isinstance(output, list) and len(output) == 1:
            output = output[0]
        
        if return_values:
            return output
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

    @property
    def eval_engine(self):
        """Get the evaluation engine for advanced use cases."""
        return self._eval_engine
    
    # ---

    def loss(self, states=None, ansatze=None, *, params=None, probabilities=None, batch_size=None, num_samples=None, num_chains=None, return_stats=True, return_values=False, **kwargs) -> Union[NQSLoss, Array]:
        '''
        Compute loss using the evaluation engine.
        
        Parameters:
        -----------
            states: 
                Array of state configurations
            ansatze: 
                Array of log ansatz values for the states
            params: 
                Optional network parameters
            probabilities: 
                Optional probability weights
            batch_size: 
                Optional batch size override
            num_samples: 
                Number of samples to generate if using the sampler
            num_chains: 
                Number of chains to use if using the sampler
            return_stats: 
                Whether to return statistics
            return_values: 
                Whether to return values only
        '''
        loss_funct                  = self.loss_function
        if params is None:          params = self._net.get_params()
        if batch_size is None:      batch_size = self._batch_size
        return self._eval_engine.loss(states=states, ansatze=ansatze, action_func=loss_funct, params=params, probabilities=probabilities, batch_size=batch_size, num_samples=num_samples, 
            num_chains=num_chains, return_stats=return_stats, return_values=return_values, **kwargs)
        
    def compute_loss(self, states=None, ansatze=None, *, params=None, probabilities=None, batch_size=None, num_samples=None, num_chains=None, return_stats=True, return_values=False, **kwargs) -> Union[NQSLoss, Array]:
        ''' Alias for the loss computation '''
        return self.loss(states=states, ansatze=ansatze, params=params, probabilities=probabilities, batch_size=batch_size, num_samples=num_samples, num_chains=num_chains, return_stats=return_stats, return_values=return_values, **kwargs)    
    
    # ---
    # specifically for energy computations    
    # ---
    
    def compute_energy(self, states=None, ansatze=None, *, params=None, probabilities=None, batch_size=None, num_samples=None, num_chains=None, return_stats=True, return_values=False, **kwargs) -> Union[NQSLoss, Array]:
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
        if self._nqsproblem.typ != 'wavefunction':
            raise ValueError(self._ERROR_ENERGY_WAVEFUNCTION)
        return self.loss(states=states, ansatze=ansatze, 
            params=params, probabilities=probabilities, batch_size=batch_size, num_samples=num_samples, 
            num_chains=num_chains, return_stats=return_stats, return_values=return_values, **kwargs)
    
    def energy(self, *args, **kwargs):          return self.compute_energy(*args, **kwargs)
    def local_energy(self, *args, **kwargs):    return self.compute_energy(*args, **kwargs)
    
    # ---
    
    def compute_observable(self, states=None, 
            ansatze=None, functions=None, names=None, *, 
            params=None, probabilities=None, batch_size=None, 
            num_samples=None, num_chains=None, return_stats=True, return_values=False, 
            args=None, **kwargs) -> Union[NQSObservable, Array]:
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
            args:
                Optional tuple of additional arguments to pass to the observable function.
                These are broadcast (not vmapped) across all sampled states.
                Useful for computing correlation functions <O_i O_j> where (i, j) are
                passed as runtime arguments, avoiding N² recompilations.

        Returns:
            ObservableResult with local values and statistics
        """
        return self._eval_engine.observable(states=states, ansatze=ansatze, functions=functions, names=names,
            params=params, probabilities=probabilities, batch_size=batch_size, num_samples=num_samples, 
            num_chains=num_chains, return_stats=return_stats, return_values=return_values, args=args, **kwargs)
            
    def expectation(self, states=None, ansatze=None, functions=None, names=None, *, params=None, probabilities=None, batch_size=None, num_samples=None, num_chains=None, return_stats=True, return_values=False, **kwargs) -> Union[NQSObservable, Array]:
        ''' Alias for the observable computation '''
        return self.compute_observable(states=states, ansatze=ansatze, functions=functions, names=names,
            params=params, probabilities=probabilities, batch_size=batch_size, num_samples=num_samples, 
            num_chains=num_chains, return_stats=return_stats, return_values=return_values, **kwargs)
    
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
            >>> (last configs, last ansatze), (all configs, all ansatze), (all probabilities) = nqs.sample(num_samples=1000, num_chains=10)
            >>> print("Sampled states:", last configs)
            >>> # Sampled states: [[...], [...], ...]
            >>> print("Sampled ansatze:", last ansatze)
            >>> # Sampled ansatze: [[...], [...], ...]
            >>> print("Sampled probabilities:", all probabilities)
            >>> # Sampled probabilities: [[...], [...], ...]        
        '''
        if reset and hasattr(self._sampler, 'reset'):
            self._sampler.reset()
        
        if params is None:
            params = self._net.get_params()
            
        (last_configs, last_ansatze), (states, all_ansatze), (all_probabilities) = self._sampler.sample(parameters=params, num_samples=num_samples, num_chains=num_chains, **kwargs)
        
        if kwargs.get('states', False):
            return states, all_ansatze
        elif kwargs.get('last', False):
            return last_configs, last_ansatze
        # default return all            
        return (last_configs, last_ansatze), (states, all_ansatze), (all_probabilities)
    
    def get_samples(self, *args, **kwargs):     return self.sample(*args, **kwargs)
    def samples(self, *args, **kwargs):         return self.sample(*args, **kwargs)
    
    @property
    def sampler(self):
        ''' Returns the sampler used for sampling the NQS. '''
        return self._sampler
    @sampler.setter
    def sampler(self, sampler: VMCSampler):
        ''' Sets a new sampler for the NQS. '''
        self.set_sampler(sampler)
    @property
    def sampler_func(self, num_samples: int = None, num_chains: int = None):
        ''' Returns the sampler function used for sampling the NQS. '''
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
    
    def _sample_for_shapes(self, ansatz_fn: AnsatzFunctionType, apply_fn: ApplyFunctionType, loss_fn: LossFunctionType, flat_grad_fn, compute_grad_f, batch_size, t=0):
        """
        Sample a configuration to get the shapes of the parameters.
        Robust version: uses try/finally to ensure sampler state is restored.
        """
        
        # Save current sampler state
        old_chains  = self._sampler.numchains
        old_samples = self._sampler.numsamples
        
        try:
            # Configure sampler for minimal run to get metadata
            # We need valid configurations to trace the JIT functions correctly
            self._sampler.set_numchains(1)
            self._sampler.set_numsamples(1)
            params = self.get_params()
            
            # Sample
            (_, _), (configs, configs_ansatze), probabilities = self._sampler.sample(params)
            
            # Run single step to infer shapes (this triggers JAX tracing)
            # We don't need the result values, just the metadata
            result = NQS._single_step(params, configs, configs_ansatze, probabilities, 
                ansatz_fn=ansatz_fn, apply_fn=apply_fn, local_loss_fn=loss_fn,
                flat_grad_fn=flat_grad_fn, compute_grad_f=compute_grad_f,
                batch_size=batch_size, t=t
            )
            
            return result.params_shapes, result.params_sizes, result.params_cpx
            
        finally:
            self._sampler.set_numchains(old_chains)
            self._sampler.set_numsamples(old_samples)

    def wrap_single_step_jax(self, batch_size: Optional[int] = None):
        """
        Wraps the single-step JAX function for use in optimization or sampling routines.
        Snapshotting: Captures the current state of ansatz functions (including modifiers).
        """
        
        batch_size                  = batch_size if batch_size is not None else self._batch_size
        self._set_batch_size(batch_size)
        
        # ! Snapshot the current functions
        # This guarantees that the trainer uses exactly what was active when wrapped
        ansatz_fn                   = self._ansatz_func
        local_loss_fn               = self._loss_func
        flat_grad_fn                = self._flat_grad_func
        apply_fn                    = self._apply_func
        compute_grad_f              = net_utils.jaxpy.compute_gradients_batched if not self._analytic else self._analytic_grad_func
        
        # ! Infer shapes safely
        shapes, sizes, iscpx        = self._sample_for_shapes(ansatz_fn, apply_fn, local_loss_fn, flat_grad_fn, compute_grad_f, batch_size=batch_size, t=0)
        
        # Pre-compute tree definition for flattening/unflattening parameters
        tree_def, flat_size, slices = self._params_tree_def, self._params_total_size, net_utils.jaxpy.prepare_slice_info(shapes, sizes, iscpx)

        # ! Bind the static arguments via partial
        # This helps JAX identify them as non-differentiable configuration
        single_step_jax             = partial(NQS._single_step, 
                                        ansatz_fn       =   ansatz_fn, 
                                        local_loss_fn   =   local_loss_fn,
                                        flat_grad_fn    =   flat_grad_fn, 
                                        apply_fn        =   apply_fn, 
                                        batch_size      =   batch_size, 
                                        compute_grad_f  =   compute_grad_f
                                    )

        # ! Create the JIT-compiled wrapper
        @jax.jit
        def wrapped(y, t, configs, configs_ansatze, probabilities, int_step=0):
            # 1. Reconstruct PyTree parameters from flat vector
            params = net_utils.jaxpy.transform_flat_params_jit(y, tree_def, slices, flat_size)
            
            # 2. Run the physics step
            result = single_step_jax(params, configs, configs_ansatze, probabilities, t=t, int_step=int_step)
            
            # 3. Return flattened results for the optimizer
            return (result.loss, result.loss_mean, result.loss_std), result.grad_flat, (shapes, sizes, iscpx)
            
        return wrapped

    @staticmethod
    def _single_step(params         : Any, 
                    configs         : Array, 
                    configs_ansatze : Any, 
                    probabilities   : Any, 
                    # functions (Static input for JIT)
                    ansatz_fn       : Callable  = None,
                    apply_fn        : Callable  = None,
                    local_loss_fn   : Callable  = None,
                    flat_grad_fn    : Callable  = None,
                    compute_grad_f  : Callable  = net_utils.jaxpy.compute_gradients_batched,
                    # Static for evaluation
                    batch_size      : int       = None,
                    t               : float     = None,
                    int_step        : int       = 0) -> NQSSingleStepResult:
        '''
        Perform a single training step. Pure function (no self access) for JAX compatibility.
        '''

        #! a) Compute Local Loss
        # For example, E_loc = <s|H|psi> / <s|psi>
        (v, means, stds) = apply_fn(
                            func            = local_loss_fn,
                            states          = configs,
                            sample_probas   = probabilities,
                            logprobas_in    = configs_ansatze,
                            logproba_fun    = ansatz_fn,
                            parameters      = params,
                            batch_size      = batch_size
                        )
        
        #! b) Compute Gradients
        # O_k = nabla log psi
        # The dtype of flat_grads depends on the network (Complex for Quantum)
        flat_grads, shapes, sizes, iscpx = compute_grad_f(
                                            net_apply                   = ansatz_fn,
                                            params                      = params,
                                            states                      = configs,
                                            single_sample_flat_grad_fun = flat_grad_fn,
                                            batch_size                  = batch_size
                                        )

        return NQSSingleStepResult(loss=v, loss_mean=means, loss_std=stds, grad_flat=flat_grads, params_shapes=shapes, params_sizes=sizes, params_cpx=iscpx)

    def step(self, 
            configs             : Array = None, 
            configs_ansatze     : Any = None, 
            probabilities       : Any = None, 
            params              : Any = None, **kwargs) -> NQSSingleStepResult:
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
        loss_function       = kwargs.get('loss_function',       self._loss_func)
        flat_grad_fun       = kwargs.get('flat_grad_fun',       self._flat_grad_func)
        compute_grad_fun    = net_utils.jaxpy.compute_gradients_batched if self._isjax else None
        
        # check if the configurations are provided
        if configs is None or configs_ansatze is None or probabilities is None:
            num_samples     = kwargs.get('num_samples', self._sampler.numsamples)
            num_chains      = kwargs.get('num_chains',  self._sampler.numchains)
            (_, _), (configs, configs_ansatze), probabilities = self._sampler.sample(parameters=params, 
                                                            num_samples=num_samples, num_chains=num_chains)
        
        
        # call the appropriate single step function
        if isinstance(self._nqsproblem, WavefunctionPhysics):
            return self._single_step(params         =   params, 
                                    configs         =   configs, 
                                    configs_ansatze =   configs_ansatze, 
                                    probabilities   =   probabilities, 
                                    ansatz_fn       =   ansatz_fn,
                                    apply_fn        =   apply_fn,
                                    local_loss_fn   =   loss_function,
                                    flat_grad_fn    =   flat_grad_fun,
                                    compute_grad_f  =   compute_grad_fun,
                                    batch_size      =   self._batch_size,
                                    **kwargs)
            
        elif isinstance(self._nqsproblem, DensityMatrixPhysics):
            raise NotImplementedError("Time evolution not implemented yet.")
        else:
            raise ValueError("Unknown problem type.")
    
    #####################################
    #! STATE MODIFIER
    #####################################
    
    @property
    def modifier(self) -> Union[Operator, Callable]:
        '''
        Return the current state modifier wrapper (AnsatzModifier).
        '''
        return self._modifier_wrapper

    @property
    def modifier_source(self) -> Union[Operator, Callable]:
        '''
        Return the original operator object used to create the modifier.
        '''
        return self._modifier_source    
    
    @property
    def modified(self) -> bool:
        '''
        Return True if the state is currently modified.
        '''
        return self._modifier_wrapper is not None
    
    def unset_modifier(self):
        """
        Unset the state modifier and restore the original ansatz function.
        """
        if self._modifier_wrapper is None:  return

        # Clear references
        self._modifier_wrapper              = None
        self._modifier_source               = None
        
        # Restore the original pure network function (Zero overhead)
        self._ansatz_func                   = self._ansatz_base_func
        
        self.log("State modifier unset. Reverted to ground state ansatz.", lvl=1, color='blue')    
    
    def set_modifier(self, modifier: Union[Operator, Callable], **kwargs):
        r"""
        Apply a linear operator $\hat{O}$ to the ansatz state: $|\tilde{\Psi}\rangle = \hat{O} |\Psi_\theta\rangle$.

        This transforms the NQS evaluation into a modified state evaluation using the 
        robust `AnsatzModifier` wrapper. It handles both diagonal (M=1) and 
        sparse/branching (M>1) operators automatically.

        .. math::
            \log \tilde{\Psi}(s) = \log \left( \sum_{k=1}^M w_k \cdot \exp(\log \Psi_\theta(s'_k)) \right)

        Parameters
        ----------
        modifier : Union[Operator, Callable]
            An operator or function that takes a state $s$ and returns:
            - `connected_states` (batch, M, N)
            - `weights` (batch, M)
        """
        
        # Ensure we are using JAX (Modifiers rely on JIT/PyTrees)
        if not self._isjax:
            raise NotImplementedError("State modifiers are currently only supported for the JAX backend.")

        self.log("Initializing State Modifier...", lvl=1)

        try:
            # We use the _ansatz_base_func to ensure we are wrapping the pure network
            wrapper = AnsatzModifier(
                net_apply           = self._ansatz_base_func,
                operator            = modifier,
                input_shape         = self._shape if self._shape else (self._nvisible,),
                dtype               = self._dtype
            )

            # Store references
            self._modifier_wrapper  = wrapper
            self._modifier_source   = modifier

            # JIT Compile the wrapper
            # Since AnsatzModifier is a registered PyTree, we can JIT it directly.
            # This creates a new high-performance executable.
            self._ansatz_func = jax.jit(wrapper)

            # Log success
            self.log(f"Modifier active. Branching factor M={wrapper.branching_factor}", lvl=1, color='green')

        except Exception as e:
            self.log(f"Failed to set modifier: {e}", lvl=0, color='red')
            raise e
    
    #####################################
    #! WEIGHTS
    #####################################
    
    def save_weights(self, step: int = 0, filename: Optional[str] = None, metadata: Optional[dict] = None):
        """
        Delegates saving to the CheckpointManager.
        
        Parameters
        ----------
        step : int
            The current training step or epoch. Used for versioning the saved weights.
        filename : Optional[str]
            The filename to save the weights to. Can be:
            - Absolute path: used directly
            - Relative path: resolved relative to self.defdir
            - None: uses default naming scheme (checkpoint_{step}.h5 or Orbax step-based)
        metadata : Optional[dict]
            Additional metadata to save with the weights.
            Seed, backend, and network type are automatically added.
            
        Returns
        -------
        Path : The path where the checkpoint was saved.
        
        Example:
        --------
            >>> nqs = NQS(model, net, sampler)
            >>> nqs.save_weights(step=10, filename="best_model.h5")
            >>> nqs.save_weights(step=100) # Uses default naming
        """
        params = self.get_params()
        
        # Build comprehensive metadata
        if metadata is None: 
            metadata = {}
        
        # Core metadata - always included
        metadata.update({
            'backend'       : self._backend_str, 
            'net_type'      : type(self._net).__name__,
            'seed'          : getattr(self, '_net_seed', self._seed),
            'shape'         : list(self.shape) if hasattr(self, 'shape') else None,
            'dtype'         : str(self._dtype),
        })
        
        # Add model info if available
        if hasattr(self, '_model') and self._model is not None:
            metadata['model'] = str(self._model)
            if hasattr(self._model, 'lattice') and self._model.lattice is not None:
                metadata['lattice'] = str(self._model.lattice)
        
        # Add exact info if available
        if self.exact is not None:
            metadata['exact'] = self.exact

        saved_path = self.ckpt_manager.save(
            step        = step if isinstance(step, int) else 'final', 
            params      = params, 
            metadata    = metadata, 
            filename    = filename
        )
        
        self.log(f"Saved weights for step {step} to {saved_path}", lvl=1, color='green')
        return saved_path

    def load_weights(self, step: Optional[int] = None, filename: Optional[str] = None):
        """
        Delegates loading to the CheckpointManager.
        
        Parameters
        ----------
        step : Optional[int]
            The training step to load. If None:
            - Orbax: loads the latest checkpoint
            - HDF5: requires filename or finds latest checkpoint_*.h5
        filename : Optional[str]
            Custom filename to load. Can be:
            - Absolute path: used directly
            - Relative path: resolved relative to self.defdir
            - None: uses step-based naming
            
        Returns
        -------
        None
        
        Example:
        --------
            >>> nqs.load_weights(step=100)
            >>> nqs.load_weights(filename='best_model.h5')
            >>> nqs.load_weights()  # Latest checkpoint
        """
        try:
            params = self.ckpt_manager.load(
                step        = step if isinstance(step, int) else 'final' if step is not None else None, 
                filename    = filename,  
                target_par  = self.get_params()
            )
            self.set_params(params)
            
            # Load metadata and set exact info
            metadata = self.ckpt_manager.load_metadata(step=step if isinstance(step, int) else 'final', filename=filename)
            if 'exact' in metadata:
                self.exact = metadata['exact']
                
            self.log(f"Loaded weights (step={step}, file={filename})", lvl=1)
        except Exception as e:
            self.log(f"Failed to load weights: {e}", lvl=0, color='red')
            raise e

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
    
    @property
    def parameters(self) -> Any:
        """Alias for get_params() - returns current network parameters."""
        return self.get_params()
    
    @parameters.setter
    def parameters(self, new_params: Any):
        """Alias for set_params() - sets new network parameters."""
        self.set_params(new_params)
    
    #####################################
    #! GETTERS AND PROPERTIES
    #####################################
    
    @property
    def net(self):                          return self._net
    @net.setter
    def net(self, new_net):
        """
        Safety wrapper: If network changes, ensure Sampler knows about it.
        """
        self._net = new_net
        
        # If the sampler has a reference to the network (ARSampler), update it
        if hasattr(self._sampler, '_net'):
            self._sampler._net = new_net
            
            # ARSampler might need to re-jit if the static apply fn changed
            if hasattr(self._sampler, '_sample_jit'):
                # Force re-creation of JIT kernel on next sample call
                # (Assuming you implement a check or just re-init the sampler)
                pass 
                
        self.log("Network updated. Sampler reference updated.", lvl=1)
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
    def model(self):
        '''
        Return the underlying physical model. For example, the Hamiltonian.
        '''
        return self._model
    
    @model.setter
    def model(self, new_model):
        '''
        Set a new physical model and reconfigure the physics/loss functions.
        
        This is useful for impurity studies where you want to reuse the same
        trained network with a modified Hamiltonian (e.g., adding impurities).
        
        Parameters:
            new_model: The new Hamiltonian/model to use.
        '''
        self._model = new_model
        self._nqsproblem.setup(self._model, self._net)
        
        # Reconfigure loss function for wavefunction problem
        if self._nqsproblem.typ == 'wavefunction':
            self._local_en_func = getattr(self._nqsproblem, "local_energy_fn", None)
            self._loss_func     = self._local_en_func
        
        # Re-initialize directory for new model
        self._init_directory()
        
        # Reset checkpoint manager for new directory
        self.ckpt_manager       = NQSCheckpointManager(
                                    directory       = self._dir_detailed, 
                                    use_orbax       = getattr(self.ckpt_manager, 'use_orbax', True), 
                                    max_to_keep     = getattr(self.ckpt_manager, 'max_to_keep', 3), 
                                    logger          = self._logger
                                )
        
        self._logger.info(f"Model updated to: {new_model}", lvl=1, color='blue')
    
    @property
    def operators(self) -> 'OperatorModule':
        '''
        Return the list of operators associated with the physical model.
        '''
        return self._model.operators if self._model is not None else None

    # ---
    
    @property
    def num_params(self):
        ''' Return the number of parameters in the neural network. '''
        return self._params_total_size
    @property
    def npar(self):
        return self._params_total_size    
    @property
    def nvisible(self):
        ''' Return the number of visible units in the neural network. '''
        return self._nvisible
    @property
    def nvis(self):
        return self._nvisible
    @property
    def size(self):
        ''' Return the size of the neural network. '''
        return self._size
    
    @property
    def batch_size(self):
        ''' Return the batch size used for sampling. '''
        return self._batch_size
    @batch_size.setter
    def batch_size(self, new_batch_size: int):
        ''' Set a new batch size for sampling. '''
        self._set_batch_size(new_batch_size)
    
    @property
    def backend(self):
        ''' Return the backend used for the neural network. '''
        return self._backend
    @property
    def nqsbackend(self):
        '''
        Return the backend used for the neural network.
        '''
        return self._nqsbackend
    @property
    def backend_str(self):
        ''' Return the backend as a string. '''
        return self._backend_str
    
    # ---
    
    @property
    def loss_function(self):
        ''' Return the loss function used for training. '''
        if self._nqsproblem.typ == 'wavefunction' or self._nqsproblem.typ == 'ground':
            return self._local_en_func
        else:
            raise NotImplementedError("Loss function not implemented for this problem type.")
        # elif self._nqsproblem.typ == 'densitymatrix' or self._nqsproblem.typ == 'time':
            # return self._tdvp_func
    
    #! Aliases for local energy
    @property
    def local_energy(self):
        ''' Return the local energy function. '''
        return self._local_en_func
    @property
    def loc_energy(self):
        ''' Alias for local_energy '''
        return self._local_en_func
    @property
    def local_en(self):
        ''' Alias for local_energy '''
        return self._local_en_func
    
    #! Other loss functions (TODO: implement)
    # ...
    
    @property
    def beta_penalty(self):
        return self._beta_penalty
    
    #####################################

    def clone(self):
        ''' Clone the NQS solver. '''
        return NQS(self._net.clone(), self._sampler.clone(), self._backend, **self._kwargs)
    
    def swap(self, other):
        ''' Swap the NQS solver with another one. '''
        return super().swap(other)
    
    #####################################
    #! TRAIN
    #####################################
    
    _trainer: Optional["NQSTrainer"] = None
    
    @property
    def trainer(self) -> Optional["NQSTrainer"]:
        """
        Returns the current NQSTrainer instance if one has been created.
        
        Returns
        -------
        NQSTrainer or None
            The trainer instance, or None if train() hasn't been called yet.
        """
        
        if self._trainer is None:
            from QES.NQS.src.nqs_train import NQSTrainer
            self._trainer = NQSTrainer(nqs = self)
            
        return self._trainer
    
    def train(self,
            n_epochs            : int                       = 300,
            checkpoint_every    : int                       = 50,
            load_checkpoint     : bool                      = False,
            *,
            # Trainer configuration (only used if override=True or no trainer exists)
            checkpoint_step     : Union[int, str]           = None,
            reset_weights       : bool                      = False,
            override            : bool                      = True,
            # Solvers
            lin_solver          : Union[str, Callable]      = 'minres_qlp',
            pre_solver          : Union[str, Callable]      = 'jacobi',
            ode_solver          : Union[str, Any]           = 'Euler',
            tdvp                : Any                       = None,
            grad_clip           : Optional[float]           = None,
            # Configuration
            n_batch             : int                       = 128,
            n_update            : Optional[int]             = None,
            num_samples         : Optional[int]             = None,
            num_chains          : Optional[int]             = None,
            num_thermal         : Optional[int]             = None,
            num_sweep           : Optional[int]             = None,
            pt_betas            : Optional[List[float]]     = None,    
            # Learning Rate and Phase Scheduling
            phases              : Union[str, tuple]         = 'default',
            # Utilities
            timing_mode         : str                       = 'detailed',
            early_stopper       : Any                       = None,
            lower_states        : List["NQS"]               = None,
            # Schedulers
            lr_scheduler        : Optional[Callable]        = None,
            reg_scheduler       : Optional[Callable]        = None,
            diag_scheduler      : Optional[Callable]        = None,
            lr                  : Optional[float]           = None,
            reg                 : Optional[float]           = None,
            # Linear Solver options
            lin_sigma           : float                     = None,
            lin_is_gram         : bool                      = True,
            lin_force_mat       : bool                      = False,
            # TDVP options
            use_sr              : bool                      = True,
            use_minsr           : bool                      = False,
            rhs_prefactor       : float                     = -1.0,
            # Diagonal shift
            diag_shift          : float                     = 1e-1,    
            # Training options
            save_path           : str                       = None,
            reset_stats         : bool                      = True,
            use_pbar            : bool                      = True,
            # Some exact solutions
            exact_predictions   : Any                       = None,
            exact_method        : str                       = None,
            # Sampler Updates
            upd_fun             : Optional[Union[str, Any]] = None,
            update_kwargs       : Optional[dict]            = None,
            **kwargs
        ) -> "NQSTrainStats":
        """
        Train the NQS using the NQSTrainer.

        It creates (or reuses) an NQSTrainer instance and runs the training loop, supporting
        advanced features such as learning rate scheduling, regularization, Stochastic Reconfiguration (SR),
        MinSR, TDVP, checkpointing, and exact solution comparison.

        Parameters
        ----------
        n_epochs : int, default=300
            Number of training epochs.

        checkpoint_every : int, default=50
            Save checkpoint every N epochs.

        reset_weights : bool, default=False
            If True, reinitialize network parameters before training.

        override : bool, default=True
            If True, always create a new trainer with the provided arguments.
            If False, reuse existing trainer if one exists (ignores other config args).

        lin_solver : str or Callable, default='scipy_cg'
            Linear solver for SR equations. Options: 'scipy_cg', 'jax_cg', custom callable.

        pre_solver : str or Callable, optional
            Preconditioner for linear solver.

        ode_solver : str, default='Euler'
            ODE integrator for parameter updates. Options: 'Euler', 'RK4', etc.

        tdvp : Any, optional
            TDVP configuration or callable.

        n_batch : int, default=128
            Batch size for VMC sampling.

        phases : str or tuple, default='default'
            Phase scheduling preset or custom phases.

        timing_mode : str, default='detailed'
            Timing mode for profiling ('detailed', 'minimal').

        early_stopper : Any, optional
            Early stopping callback or configuration.

        lower_states : List[NQS], optional
            List of lower-energy NQS states for orthogonalization (excited states).

        lr_scheduler : str or Callable, optional
            Learning rate scheduler type or instance.
            Options: 'constant', 'exponential', 'cosine', 'linear', 'adaptive', 'step'.
            Scheduler kwargs can be passed via **kwargs:
                - lr_init, lr_final, lr_decay_rate, lr_patience, lr_min_delta, lr_cooldown, lr_step_size

        reg : float, optional
            Regularization strength.

        reg_scheduler : str or Callable, optional
            Regularization scheduler for SR matrix.

        diag_shift : float, default=1e-5
            Diagonal regularization for SR matrix.

        diag_scheduler : str or Callable, optional
            Diagonal shift scheduler for SR matrix.

        use_sr : bool, default=True
            Use Stochastic Reconfiguration (SR) for optimization.

        use_minsr : bool, default=False
            Use MinSR (memory efficient SR).

        rhs_prefactor : float, default=-1.0
            Prefactor for the TDVP right-hand side.

        save_path : str, optional
            Directory or filename for saving checkpoints.

        reset_stats : bool, default=True
            If True, reset training statistics before starting.

        use_pbar : bool, default=True
            Show progress bar during training.

        exact_predictions : array-like, optional
            Exact reference values (e.g., from ED) for comparison.

        exact_method : str, optional
            Method used to compute exact predictions (e.g., 'lanczos').

        lin_force_mat : bool, default=False
            Force forming full matrix in linear solver.

        lin_is_gram : bool, default=True
            Treat linear system as Gram matrix (S) if True, else covariance.

        p_global : float, optional
            Probability (0.0 to 1.0) of proposing a global update in the sampler (default: 0.0).
            See `VMCSampler` documentation for details.
        global_fraction : float, optional
            Fraction of spins to flip during a global update (default: 0.5).
            See `VMCSampler` documentation for details.
            
        upd_fun : str or Enum, optional
            Update rule for the sampler (e.g., "LOCAL", "EXCHANGE").
            If provided, reconfigures the sampler's update function before training.
            
        update_kwargs : dict, optional
            Additional arguments for the update rule (e.g., {'patterns': [...]}).

        **kwargs
            Additional arguments passed to NQSTrainer and schedulers.

        Returns
        -------
        NQSTrainStats
            Training statistics including loss history, timing, checkpoints, etc.

        Examples
        --------
        >>> # Basic training with default settings
        >>> stats = psi.train(n_epochs=200)

        >>> # Training with cosine annealing learning rate scheduler
        >>> stats = psi.train(n_epochs=500, lr=1e-3, lr_scheduler='cosine', min_lr=1e-5)

        >>> # Training with exact diagonalization comparison
        >>> hamil.diagonalize()
        >>> stats = psi.train(n_epochs=300, exact_predictions=hamil.eigenvalues, exact_method='lanczos')

        >>> # Continue training with same optimizer state
        >>> stats = psi.train(n_epochs=100)
        >>> stats = psi.train(n_epochs=100, override=False)

        >>> # Advanced: MinSR, custom batch size, checkpointing
        >>> stats = psi.train(n_epochs=1000, use_minsr=True, n_batch=2048, checkpoint_every=100, save_path='./checkpoints')
        
        >>> # Change update rule for training
        >>> stats = psi.train(n_epochs=100, upd_fun="EXCHANGE")

        See Also
        --------
        NQSTrainer : 
            Full trainer class with more configuration options.
        NQS.help('train') : 
            Interactive help and usage tips.
        """

        # Import here to avoid circular imports

        if reset_weights:
            self.reset()
            self.log("Network parameters reset before training.", lvl=1, color='blue')

        if self._sampler.name == 'VMC':
            if num_samples is not None:
                self._sampler.set_numsamples(num_samples)
            if num_chains is not None:
                self._sampler.set_numchains(num_chains)
            if num_thermal is not None and hasattr(self._sampler, 'set_therm_steps'):
                self._sampler.set_therm_steps(num_thermal)
            if num_sweep is not None and hasattr(self._sampler, 'set_sweep_steps'):
                self._sampler.set_sweep_steps(num_sweep)
            if pt_betas is not None and hasattr(self._sampler, 'set_pt_betas'):
                self._sampler.set_pt_betas(pt_betas)
            
            # Configure Global Updates if parameters are present in kwargs
            if 'global_p' in kwargs or 'p_global' in kwargs or 'global_fraction' in kwargs or 'global_update' in kwargs:
                if hasattr(self._sampler, 'set_global_update'):
                    # Support both names for backward compatibility
                    g_p = kwargs.get('global_p', kwargs.get('p_global', 0.0))
                    self._sampler.set_global_update(global_p = g_p, global_fraction = kwargs.get('global_fraction', 0.5), global_update = kwargs.get('global_update', None))
                    if g_p > 0:
                        self.log(f"Global sampler update set to: {g_p:.2f} ({kwargs.get('global_update', 'default')})", lvl=2, color='blue')
            
            # Configure Update Function
            if upd_fun is not None and hasattr(self._sampler, 'set_update_function'):
                u_kwargs = update_kwargs or {}
                u_kwargs.update({'hilbert' : u_kwargs.pop('hilbert', self._model.hilbert), 'lattice': self._model.lattice})
                self._sampler.set_update_function(upd_fun, **u_kwargs)
                self.log(f"Sampler update function set to: {upd_fun}", lvl=2, color='blue')
        
            if any(param is not None for param in [num_samples, num_chains, num_thermal, num_sweep, pt_betas]):
                self.log(f"Sampler (re)configured: num_samples={self._sampler.numsamples}, num_chains={self._sampler.numchains}", lvl=2, color='blue')
                if pt_betas is not None and hasattr(self._sampler, 'pt_betas'):
                    self.log(f"  PT Betas: {self._sampler.pt_betas}", lvl=2, color='blue')
            
        # Create or reuse trainer
        if override or self._trainer is None:
            from QES.NQS.src.nqs_train import NQSTrainer
            
            old_stats       = self._trainer.stats if self._trainer is not None else None

            self._trainer   = NQSTrainer(
                nqs             = self,
                # Solvers
                lin_solver      = lin_solver,
                lin_force_mat   = lin_force_mat,
                pre_solver      = pre_solver,
                ode_solver      = ode_solver,
                tdvp            = tdvp,
                # Configuration
                n_batch         = n_batch,
                phases          = phases,
                # Utilities
                timing_mode     = timing_mode,
                early_stopper   = early_stopper,
                logger          = self._logger,
                lower_states    = lower_states,
                # Schedulers
                lr_scheduler    = lr_scheduler,
                lr_max_epochs   = kwargs.pop('lr_max_epochs', n_epochs),
                reg_scheduler   = reg_scheduler,
                reg_max_epochs  = kwargs.pop('reg_max_epochs', n_epochs),
                diag_scheduler  = diag_scheduler,
                diag_max_epochs = kwargs.pop('diag_max_epochs', n_epochs),
                # Training options
                lr              = lr,
                reg             = reg,
                diag_shift      = diag_shift,
                # Linear Solver
                lin_sigma       = lin_sigma,
                lin_is_gram     = lin_is_gram,
                # TDVP
                use_sr          = use_sr,
                use_minsr       = use_minsr,
                rhs_prefactor   = rhs_prefactor,
                grad_clip       = grad_clip,
                **kwargs
            )
            
            if not reset_stats and old_stats is not None:
                self._trainer.stats = old_stats

        # Updates in sampler
        if hasattr(self._sampler, 'set_update_num') and n_update is not None:
            self._sampler.set_update_num(n_update)
            self.log(f"Sampler update number set to {n_update}", lvl=2, color='blue')

        if exact_predictions is None and self._model.eig_vals is not None:
            exact_predictions   = self._model.eig_vals
            self.exact          = exact_predictions

        if self.exact and exact_predictions is None:
            exact_predictions = self.exact.get('exact_predictions', exact_predictions)
            exact_method      = self.exact.get('exact_method',      exact_method)
            
        # Run training
        if load_checkpoint:
            try:
                # Load the state (weights + stats)
                stats                   = self._trainer.load_checkpoint(step=checkpoint_step)
                stats.exact_predictions = exact_predictions
                reset_stats             = False
                return stats
            
            except Exception as e:
                self.log(f"Requested checkpoint load failed: {e}", lvl=0, color='red', log='warning')
                raise e

            
        stats = self._trainer.train(
            n_epochs            = n_epochs,
            checkpoint_every    = checkpoint_every,
            save_path           = save_path,
            reset_stats         = reset_stats or reset_weights,
            use_pbar            = use_pbar,
            exact_predictions   = exact_predictions,
            exact_method        = exact_method,
            **kwargs
        )
        
        return stats
    
    #####################################
    
    def __repr__(self):
        return f"NQS(logansatz={self._net},sampler={self._sampler},backend={self._backend_str},mod={self.modified})"
    
    def __str__(self):
        return self.__repr__() 

    def help(self, topic: str = "general"):
        """
        Prints usage information and physics background for NQS features.
        
        Parameters
        ----------
        topic : str
            The topic to query. Options:
            - 'general': 
                Overview of the NQS object.
            - 'modifier':
                Details on state modifiers (Projectors, Symmetries).
            - 'sampling': 
                Info on VMC vs Autoregressive sampling.
            - 'network': 
                Details about the loaded ansatz.
            - 'networks':
                Overview of available ansatz architectures.
            - 'usage': 
                Example workflows and common operations.
            - 'train':
                Training with the train() method (LR scheduling, SR, etc.).
            - 'checkpoints':
                Saving and loading model weights and checkpoints.
        """
        topic   = topic.lower().strip()
        msg     = ""
        border  = "-" * 60
        
        if topic == "general":
            msg = f"""
                {border}
                NQS Solver Help: General
                {border}
                This object represents a variational quantum state |Psi_theta>.
                
                Current Configuration:
                - Backend: {self._backend_str}
                - Ansatz: {type(self._net).__name__}
                - Params: {self.num_params}
                - Sampler: {type(self._sampler).__name__} (Batch: {self._batch_size}, mu: {self._sampler._mu})
                
                Key Methods:
                - train(...): Optimize parameters via VMC/TDVP.
                - sample(...): Generate configurations s ~ |Psi(s)|^{self._sampler._mu} (default mu=2 - squared amplitude).
                - evaluate(s): Compute log(Psi(s)).
                - set_modifier(O): Transform ansatz to O|Psi>.
                """
        
        elif topic == "modifier":
            msg = f"""
                {border}
                NQS Solver Help: State Modifiers
                {border}
                Modifiers allow you to apply an operator O to the ansatz:
                |Psi_new> = O |Psi_old>
                
                This is done on-the-fly during evaluation. The NQS computes:
                log <s|Psi_new> = log ( sum_k <s|O|s'_k> * <s'_k|Psi_old> )
                
                Usage:
                    op = Operator(...) # e.g., Symmetry projector or S^z
                    nqs.set_modifier(op)
                
                Performance Modes (Auto-Detected):
                1. Single-Branch (M=1):
                Operator maps s -> s'. Very fast. 
                Used for: 
                    Quantum Numbers (S^z, N), Basis Rotations. Depends on operator.
                
                2. Multi-Branch (M>1):
                Operator maps s -> sum_k w_k |s'_k>. Slower (evaluates net M times).
                Used for: 
                    Symmetries (Sum over group), Hamiltonian action.
                
                Current Status:
                - Modified: {self.modified}
                - Modifier: {self._modifier}
                """
            
        elif topic == "sampling":
            msg = f"""
                {border}
                NQS Solver Help: Sampling
                {border}
                Current Sampler: {type(self._sampler).__name__}
                
                1. MCMC (VMCSampler):
                - Uses Metropolis-Hastings.
                - Good for general RBMs/CNNs.
                - Suffers from autocorrelation time (tau).
                
                2. Autoregressive (ARSampler):
                - Generates samples sequentially (s1 -> s2 -> ...).
                - Zero autocorrelation (iid samples).
                - Requires 'ar' or 'made' network architecture.
                - Exact likelihoods P(s) available.
                
                Update Rules (VMC):
                - "LOCAL" (default): 
                    - Single spin flips.
                - "EXCHANGE": 
                    - Neighbor swaps (conserves N, Sz).
                - "GLOBAL": 
                    - Pattern/Plaquette flips (reduces autocorrelation).
                - "MULTI_FLIP": 
                    - Flips N random sites.
                
                Change rule: 
                    psi = NQS(..., upd_fun="EXCHANGE", hilbert=h)
                    # or during training
                    psi.train(..., upd_fun="GLOBAL", update_kwargs={'patterns': [...]})
                """
        
        elif topic == "usage":
            msg = f"""
                {border}
                NQS Solver Help: General usage
                {border}
                Current Ansatz: {type(self._net).__name__}

                1. Initialization
                psi     = NQS(logansatz='ar', model=hamil, sampler='ARSampler')
                # or 
                psi     = NQS(logansatz=custom_net, sampler=custom_sampler)
                # or VMC
                psi     = NQS(logansatz='rbm', model=hamil, sampler='MCSampler', backend='jax', s_numsamples=5000)

                2. Training (via NQSTrainer)
                trainer = NQSTrainer(psi, lin_solver='jax_cg', ...)
                stats   = trainer.train(n_epochs=100)

                3. Sampling & Observables
                # Get raw samples and log-amplitudes
                (_, _), (configs, log_psi), weights = psi.sample(num_samples=1000)
                
                # Compute Expectation Values (e.g. Energy)
                E_stats = psi.compute_energy(configs)
                print(f"Energy: {{E_stats.mean:.4f}} +/- {{E_stats.error:.4f}}")

                4. I/O Operations
                psi.save_weights("checkpoint.h5")
                psi.load_weights("checkpoint.h5")

                5. Dynamic Settings (What you can change)
                psi.batch_size = 2048  # Adjust batch size for evaluation
                psi.net = new_net      # Swap architecture (resets optimizer)
                psi.sampler = 'vmc'    # Switch sampling strategy
                
                6. Compute observables, for example for Hamiltonian for spin-1/2
                lat = Lattice(...)
                mod = Hamiltonian(...)
                sig_x = mod.operators.sig_x(lattice=lat, sites=[0])
                obs_x = psi.compute_observable(sig_x, num_samples=1000)
                print(f"<Sx_0> = {{obs_x.mean:.4f}} +/- {{obs_x.error:.4f}}")
                """
        
        elif topic == "train":
            msg = f"""
                {border}
                NQS Solver Help: Training with train()
                {border}
                The train() method provides a convenient way to train the NQS
                without manually creating an NQSTrainer instance.
                
                Basic Usage:
                    stats = psi.train(n_epochs=300)
                
                Key Parameters:
                - n_epochs: Number of training epochs (default: 300)
                - checkpoint_every: Save checkpoint every N epochs (default: 50)
                - override: If True (default), create new trainer; if False, reuse existing
                
                Learning Rate Scheduling:
                    # Constant LR
                    stats = psi.train(lr=1e-3)
                    
                    # Cosine annealing (recommended)
                    stats = psi.train(lr=1e-2, lr_scheduler='cosine', min_lr=1e-5)
                    
                    # Exponential decay (gamma^epoch)
                    stats = psi.train(lr=1e-2, lr_scheduler='exponential', lr_decay=0.99)
                
                Stochastic Reconfiguration:
                    # Standard SR
                    stats = psi.train(use_sr=True, diag_shift=1e-4)
                    
                    # MinSR (memory efficient for large networks)
                    stats = psi.train(use_minsr=True)
                    
                    # Plain gradient descent
                    stats = psi.train(use_sr=False)
                
                Comparison with Exact Diagonalization:
                    hamil.diagonalize()
                    stats = psi.train(exact_predictions=hamil.eigenvalues, exact_method='lanczos')
                
                Continuing Training (reuse optimizer state):
                    stats = psi.train(n_epochs=100)         # First run
                    stats = psi.train(n_epochs=100, override=False)  # Continue
                
                Accessing the Trainer:
                    psi.train(n_epochs=100)
                    trainer = psi.trainer  # Access underlying NQSTrainer
                    
                Current Status:
                - Trainer exists: {self._trainer is not None}
                - Trainer type: {type(self._trainer).__name__ if self._trainer else 'None'}
                """
        
        elif topic == "network":
            info = NetworkFactory.net_help()
            if self._net is not None:
                net_type    = self._net.name if hasattr(self._net, 'name') else type(self._net).__name__
                msg         = f"""
                            {border}
                            NQS Solver Help: Network Details
                            {border}
                            Current Network Type: {net_type}
                            
                            {info.get(net_type, "No additional info available for this network type.")}
                            """
        elif topic == "networks":
            info = NetworkFactory.net_help()
            msg  = f"""
                {border}
                NQS Solver Help: Available Networks
                {border}
                The following ansatz architectures are available:
                
                {chr(10).join([f"- {k}: {v.splitlines()[0]}" for k,v in info.items()])}
                
                For detailed information on each architecture, use:
                    psi.help(topic='network')
                """
                
        elif topic == "checkpoints":
            msg = f"""
                {border}
                NQS Solver Help: Checkpoints & I/O
                {border}
                Save and load model weights for resuming training or inference.
                
                Manual Save/Load (Weights Only):
                    # Save current network parameters
                    psi.save_weights("my_model.h5")
                    
                    # Load weights (network architecture must match)
                    psi.load_weights("my_model.h5")
                    
                    # With custom path
                    psi.save_weights("/path/to/checkpoints/epoch_100.h5")
                
                Automatic Checkpoints During Training:
                    stats = psi.train(
                        n_epochs=300,
                        checkpoint_every=50,       # Save every 50 epochs
                        save_path="./checkpoints"  # Directory for auto-saves
                    )
                    # Creates: ./checkpoints/epoch_50.h5, epoch_100.h5, ...
                
                Resume Training from Checkpoint:
                    # Load weights from previous run
                    psi.load_weights("./checkpoints/epoch_200.h5")
                    
                    # Continue training (new trainer)
                    stats = psi.train(n_epochs=100)
                    
                    # Or continue with same optimizer state
                    stats = psi.train(n_epochs=100, override=False)
                
                File Format:
                    - HDF5 (.h5) format for weights
                    - Contains: network parameters, metadata
                    - Compatible with JAX/Flax serialization
                
                Tips:
                    - Always save after successful training
                    - Use descriptive filenames (e.g., 'gs_N16_E-5.234.h5')
                    - Keep checkpoint_every reasonable (50-100 epochs)
                """
        
        else:
            msg = f"Unknown topic '{topic}'. Try: 'general', 'modifier', 'sampling', 'usage', 'train', 'checkpoints'."
            
        print(msg)

    #####################################
    #! STATIC
    #####################################
    
    @staticmethod
    def _compute_transition_element(nqs_bra: 'NQS', nqs_ket: 'NQS', operator: 'Operator', num_samples: int = 4096, num_chains: int = 1, operator_args: dict = {}) -> complex:
        """
        Computes the normalized transition matrix element:
        M_12 = <Psi_bra | O | Psi_ket> / sqrt(<bra|bra> * <ket|ket>)
        
        Uses bidirectional importance sampling to handle normalization constants correctly.
        
        Args:
            nqs_bra: The state on the left <Psi_1|
            nqs_ket: The state on the right |Psi_2>
            operator: The operator O (must be applicable to nqs_ket)
            num_samples: MC samples
            num_chains: Number of Markov chains
            operator_args: Additional arguments for the operator
            
        Returns:
            The normalized expectation value.
        """
        
        # ------------------------------------------------------------------
        # Step 1: Sample from BRA distribution (Psi_1)
        # ------------------------------------------------------------------
        
        if isinstance(operator, Operator):
            operator_fun = operator.jax
        elif isinstance(operator, Callable):
            operator_fun = operator
        else:
            raise ValueError("Operator must be an instance of Operator or a callable function.")
            
        # We need samples to compute Q (Transition) and R1 (Overlap Forward)
        (_, _), (s1, log_psi1_s1), _    = nqs_bra.sample(num_samples=num_samples, num_chains=num_chains)
        
        # Evaluate Ket on Bra samples
        log_psi2_s1                     = nqs_ket.ansatz(s1)
        
        # Calculate Log Ratio: log(Psi_2(s) / Psi_1(s))
        log_ratio_12                    = log_psi2_s1 - log_psi1_s1
        
        # Compute R1: <1|2> / <1|1>
        # Shift for stability: exp(x - max)
        # R1 = mean(exp(log_ratio))
        lmax_1                          = jnp.max(jnp.real(log_ratio_12))
        R1                              = jnp.mean(jnp.exp(log_ratio_12 - lmax_1)) * jnp.exp(lmax_1)
        
        # -- Compute Q: <1|O|2> / <1|1> --
        # Q = mean( (Psi_2/Psi_1) * O_loc_2 )
        
        # Compute Local Estimator of O on state 2 at positions s1
        # nqs.apply returns (stats), but with return_values=True it usually returns raw values depending on engine
        # We assume standard behavior: returns array of local values
        # NOTE: We pass (s1, log_psi2_s1) so it doesn't re-evaluate ansatz
        loc_O_ket                       = nqs_ket.apply(operator_fun, states_and_psi=(s1, log_psi2_s1), return_values=True, args=operator_args)
        
        # If apply returned a stats object, extract values. If array, use directly.
        if hasattr(loc_O_ket, 'values'): loc_O_ket = loc_O_ket.values
        
        # Weighted average: E[ exp(log_ratio) * loc_val ]
        # Use same shift lmax_1 for stability
        Q                               = jnp.mean(jnp.exp(log_ratio_12 - lmax_1) * loc_O_ket) * jnp.exp(lmax_1)

        # ------------------------------------------------------------------
        # Step 2: Sample from KET distribution (Psi_2)
        # ------------------------------------------------------------------
        # We only need this to compute the normalization ratio (R2)
        (_, _), (s2, log_psi2_s2), _    = nqs_ket.sample(num_samples=num_samples, num_chains=num_chains)
        log_psi1_s2                     = nqs_bra.ansatz(s2)
        
        # Calculate Log Ratio: log(Psi_1(s) / Psi_2(s))
        log_ratio_21                    = log_psi1_s2 - log_psi2_s2
        
        # -- Compute R2: <2|1> / <2|2> --
        lmax_2                          = jnp.max(jnp.real(log_ratio_21))
        R2                              = jnp.mean(jnp.exp(log_ratio_21 - lmax_2)) * jnp.exp(lmax_2)
        
        # ------------------------------------------------------------------
        # Step 3: Combine for Normalized Result
        # ------------------------------------------------------------------
        
        # Handle division by zero if overlap is extremely small
        if jnp.abs(R1) < 1e-12 or jnp.abs(R2) < 1e-12:
            return 0.0 + 0.0j

        normalization_ratio             = jnp.sqrt(R2 / jnp.conj(R1))
        result                          = Q * normalization_ratio
        return result

    @staticmethod
    def compute_overlap(nqs_a: 'NQS', nqs_b: 'NQS', *, num_samples: int = 4096, num_chains: Optional[int] = None, operator: Optional[Any] = None, operator_args: Optional[Any] = None) -> float:
        r"""
        Computes the squared overlap (Fidelity) between two NQS instances:
        F = |<Psi_A | Psi_B>|^2 / (<Psi_A|Psi_A> <Psi_B|Psi_B>)
        This is done via Monte Carlo estimation using samples from both distributions.
        
        Mathematically:
        F = ( E_{s ~ |Psi_A|^2} [ Psi_B(s) / Psi_A(s) ] ) *
            ( E_{s ~ |Psi_B|^2} [ Psi_A(s) / Psi_B(s) ] )
        
        where E denotes the expectation value over the sampled configurations.
        
        Args:
            nqs_a: First NQS instance (e.g., current training step).
            nqs_b: Second NQS instance (e.g., target state or previous step).
            num_samples: Number of MC samples to use for estimation.
            num_chains: Number of independent Markov chains for sampling.
            operator: Optional operator to insert between states (not implemented).
            operator_args: Additional arguments for the operator (if any).
        
        Returns:
            float: The squared overlap (between 0.0 and 1.0).
        """
        
        if nqs_a.nvisible != nqs_b.nvisible:
            raise ValueError("NQS instances must have the same number of visible units to compute overlap.")
        
        if operator is not None:
            return _compute_transition_element(nqs_a, nqs_b, operator, num_samples, num_chains, operator_args)

        #! WORK WITH JAX ONLY    
        import jax.numpy as jnp
        import jax.scipy.special as jsp
        
        # Sample from Distribution A 
        # We ignore the 'last' samples and take 'all' samples for better statistics
        (_, _), (samples_a, log_psi_a_on_a), _  = nqs_a.sample(num_samples=num_samples, num_chains=num_chains)
        
        # Evaluate Ansatz B on samples from A
        # Note: nqs.ansatz() handles batching internally based on nqs._batch_size
        log_psi_b_on_a                          = nqs_b.ansatz(samples_a)
        
        # Compute Ratio 1: <Psi_A | Psi_B> / <Psi_A | Psi_A>
        # ratio = exp( log_psi_b - log_psi_a )
        # We use Log-Sum-Exp for numerical stability: 
        # mean(exp(x)) = exp(logsumexp(x) - log(N))
        log_ratio_1                             = log_psi_b_on_a - log_psi_a_on_a
        log_mean_1                              = jsp.logsumexp(log_ratio_1) - jnp.log(samples_a.shape[0])
        
        # Step 2: Sample from Distribution B
        (_, _), (samples_b, log_psi_b_on_b), _  = nqs_b.sample(num_samples=num_samples, num_chains=num_chains)
        
        # Evaluate Ansatz A on samples from B
        log_psi_a_on_b                          = nqs_a.ansatz(samples_b)
        
        # Compute Ratio 2: <Psi_B | Psi_A> / <Psi_B | Psi_B>
        log_ratio_2                             = log_psi_a_on_b - log_psi_b_on_b
        log_mean_2                              = jsp.logsumexp(log_ratio_2) - jnp.log(samples_b.shape[0])
        
        # Step 3: Combine 
        # F = Mean(B/A)_A * Mean(A/B)_B
        # In log space:                         log_F = log_mean_1 + log_mean_2
        # The result should be real-valued (overlap of normalized vectors).
        # We take the real part of the exponential.
        
        fidelity = jnp.exp(log_mean_1 + log_mean_2)
        return float(jnp.real(fidelity))

    #####################################
    #! SAMPLES FOR MCMC, IF APPLICABLE
    #####################################

    @staticmethod
    def est(loss: np.ndarray, last_el: float = 10):
        ''' Estimate value from last elements of loss array '''
        if isinstance(loss, (list, tuple, np.ndarray)):
            loss        = np.array(loss)
            if 0 < last_el < 1:
                n       = int(len(loss) * last_el)
                lossr   = loss[-n:]
            else:
                last_el = min(last_el, len(loss))
                lossr   = loss[-last_el:]
                
            lossv = np.mean(lossr)
        else:
            lossv = loss
        return lossv

    @staticmethod
    def relative_error(loss: np.ndarray, reference: float, last_el: float = 10):
        ''' '''
        if np.isclose(reference, 0.0):
            raise ValueError("Reference value is zero, cannot compute relative error.")
        return np.abs(NQS.est(loss, last_el=last_el) - reference) / np.abs(reference)
        
    @staticmethod
    def get_auto_config(system_size, target_total_samples=4096, dtype=jnp.complex64, logger: Logger = None, *, net_depth_estimate: int = 64, num_therm: Optional[int] = None, num_sweep: Optional[int] = None) -> dict:
        """
        Automatically detects hardware and returns optimal VMC parameters.
        
        Args:
            system_size (int): Number of spins/sites (N).
            target_total_samples (int): Total samples desired for the gradient batch.
            dtype (jax.dtype): The data type used for the network.

        Returns:
            dict: A dictionary of parameters ready to pass to NQS/VMCSampler.
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for automatic configuration. Please install JAX to use this feature.")
        
        import os
        import jax
        import jax.numpy as jnp
        import psutil
        
        logme           = lambda x: print(x) if logger is None else logger.info(x, lvl=1, color='green')
        # Detect Backend
        try:
            backend     = jax.default_backend()
            devices     = jax.devices()
            device_name = str(devices[0].device_kind) if devices else "Unknown"
        except:
            backend     = 'cpu'
            device_name = "CPU"
        
        logme(f"Auto-detected backend: {backend.upper()} ({device_name})")

        # 2. Check Precision (Critical for consumer GPUs)
        config              = {}
        is_double_precision = (dtype == jnp.complex128) or (dtype == jnp.float64)
        if backend == 'gpu' and is_double_precision:
            logme("WARNING: Double precision (complex128) detected on GPU.")
            logme("This effectively disables parallelism on consumer GPUs.")
            logme("Performance will be degraded by ~50x.")

        # =========================================================================
        # STRATEGY A: GPU (Wide & Shallow)
        # Goal      : Fill VRAM with chains, minimize sequential loops.
        # =========================================================================
        
        if backend == 'gpu':
            import subprocess
            
            def get_gpu_specs():
                """
                Queries the GPU for Model Name and Total VRAM (in MB).
                Returns defaults if GPU is not found or nvidia-smi fails.
                """
                gpu_info    = {
                                'name'              : 'Unknown GPU',
                                'total_memory_mb'   : 4096,             # Safe fallback (4GB)
                                'free_memory_mb'    : 2048
                            }
                
                # 1. Try JAX for Name
                try:
                    devices = jax.local_devices()
                    if devices and devices[0].platform == 'gpu':
                        gpu_info['name'] = devices[0].device_kind
                except:
                    pass

                # 2. Try nvidia-smi for Memory details
                try:
                    # Query total and free memory
                    cmd                         = "nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits"
                    output                      = subprocess.check_output(cmd.split()).decode('ascii').strip().split('\n')[0]
                    total, free                 = map(int, output.split(','))
                    gpu_info['total_memory_mb'] = total
                    gpu_info['free_memory_mb']  = free
                    logme(f"Detected GPU: {gpu_info['name']} with {total} MB VRAM ({free} MB free).")
                    
                except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
                    logme("Warning: Unable to query GPU memory via nvidia-smi. Using default values.")
                
                return gpu_info
            
            gpu_specs               = get_gpu_specs()
            precision_factor        = 2 if is_double_precision else 1
            fixed_overhead_mb       = 1024
            
            # Cost per chain (variable)
            # N * depth * width factors... simplified heuristic:
            # A complex64 number is 8 bytes.
            estimated_chain_mb      = (system_size * net_depth_estimate * 8) / (1024**2) 
            estimated_chain_mb      = max(0.5, estimated_chain_mb) * precision_factor

            # Available memory for batching (leaving 1GB buffer)
            available_for_batch     = max(512, gpu_specs['free_memory_mb'] - fixed_overhead_mb)
            
            # Calculate Max Chains
            calculated_chains       = int(available_for_batch / estimated_chain_mb)
            
            # Clamp to hardware reasonable limits (powers of 2)
            # Even with 24GB VRAM, going > 8192 yields diminishing returns
            max_limit               = 256 if is_double_precision else 8192
            optimal_chains          = min(calculated_chains, max_limit)
            
            # Round down to nearest power of 2
            optimal_chains          = 2 ** (optimal_chains.bit_length() - 1)
            optimal_chains          = max(32, optimal_chains) # Minimum 32
            optimal_chains          = min(target_total_samples, optimal_chains)
            num_samples_per_chain   = max(1,  target_total_samples // optimal_chains)
            
            # GPU hates sequential loops. Keep these minimal.
            config                  = {
                's_numchains'       : optimal_chains,
                's_numsamples'      : num_samples_per_chain,
                's_therm_steps'     : num_therm if num_therm is not None else 10,   # Quick re-adjustment
                's_sweep_steps'     : num_sweep if num_sweep is not None else 1,    # Reliance on ensemble averaging
                'hardware'          : f'gpu,{device_name.lower()},{"fp64" if is_double_precision else "fp32"}'
            }
            logme(f"Optimized: {optimal_chains} Parallel Chains x {num_samples_per_chain} Samples")

        # =========================================================================
        # STRATEGY B: CPU (Narrow & Deep)
        # Goal      : Match CPU cores, run long sequential loops.
        # =========================================================================
        else:
            # Detect physical cores
            physical_cores          = psutil.cpu_count(logical=False) or 4
            
            # Set chains equal to cores to avoid context switching
            optimal_chains          = physical_cores
            
            # We need to get all our data from length, not width
            num_samples_per_chain   = max(1, -(-target_total_samples // optimal_chains))

            # CPU needs to sweep to decorrelate because we have few chains
            config = {
                's_numchains'       : optimal_chains,
                's_numsamples'      : num_samples_per_chain,
                's_therm_steps'     : num_therm if num_therm is not None else max(20, system_size),      # Rule of thumb: N
                's_sweep_steps'     : num_sweep if num_sweep is not None else max(1,  system_size // 2), # Rule of thumb: N/2
                'hardware'          : 'cpu'
            }
            logme(f"Optimized: {optimal_chains} Parallel Chains x {num_samples_per_chain} Samples")

        # Calculate actual batch size generated
        total_batch = config['s_numchains'] * config['s_numsamples']
        logme(f"Total VMC Batch Size: {total_batch} samples per evaluation.")
        
        return config

#########################################
#! EOF
#########################################