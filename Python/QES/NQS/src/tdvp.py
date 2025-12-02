'''
This module implements the Time-Dependent Variational Principle (TDVP) for quantum systems.
It provides classes and methods to perform time evolution of quantum states using variational techniques.

In this module, we implement the core algorithms and data structures needed for TDVP, including:
- TDVP class: 
    Main class to handle TDVP computations.
- TDVPConfig class: 
    Configuration class for TDVP parameters - such as solver options, regularization, and backend settings.
- TDVPUtils class: 
    Utility functions for TDVP computations.

----------------------------
File    : QES/NQS/tdvp.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
Date    : 2025-11-01
----------------------------
'''

import os
import warnings
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Optional, Union, Any, List

try:
    JAX_AVAILABLE = os.environ.get('PY_JAX_AVAILABLE', '1') == '1'
    
    # Common utilities
    from QES.general_python.common.timer                        import timeit
    from QES.general_python.algebra.utils                       import get_backend, Array
    
    # Stochastic reconfiguration for TDVP
    import QES.general_python.algebra.solvers.stochastic_rcnfg  as sr
    
    # Preconditioners and Solvers for A * x = b ; used in TDVP
    import QES.general_python.algebra.preconditioners           as precond
    import QES.general_python.algebra.solvers                   as solvers
    
except ImportError:
    raise ImportError("Please install the 'general_python' package...")

#################################################################

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
else:
    jax = None
    jnp = np

#################################################################

@dataclass
class TDVPTimes:
    prepare     : float = 0.0
    gradient    : float = 0.0
    covariance  : float = 0.0
    x0          : float = 0.0
    solve       : float = 0.0
    phase       : float = 0.0
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def todict(self):
        return {
            'prepare'   : self.prepare,     # time for preparation
            'gradient'  : self.gradient,    # gradient computation time
            'covariance': self.covariance,  # covariance matrix preparation time
            'x0'        : self.x0,          # initial guess time
            'solve'     : self.solve,       # solve linear system time
            'phase'     : self.phase,       # phase estimation time
            'epoch'     : self.prepare + self.gradient + self.covariance + self.x0 + self.solve + self.phase
        }
        
@dataclass
class TDVPStepInfo:
    mean_energy     : Array
    std_energy      : Array
    # other
    failed          : Optional[bool] = None
    sr_converged    : Optional[bool] = None
    sr_executed     : Optional[bool] = None
    sr_iterations   : Optional[int]  = None
    timings         : TDVPTimes = field(default_factory=TDVPTimes)
    # Global phase tracking for dynamical correlations Santos, Schmitt, Heyl, PRL 131, 046501 (2023)
    theta0_dot      : Optional[Array] = None  # Time derivative of global phase θ̇₀
    theta0          : Optional[Array] = None  # Global phase parameter θ₀

#################################################################

@dataclass
class TDVPLowerPenalty:
    # most important
    excited_on_lower        : Array         # (n_j,) complex; ratios Psi_{W}(\sum _j)
    lower_on_excited        : Array         # (n_j,) complex; ratios Psi_{W_j}(\sigma )
    # parameters of the lower state
    lower_on_lower          : Array         # (n_j,) complex; ratios Psi_{W_j}(\sum _j)
    excited_on_excited      : Array         # (n_j,) complex; ratios Psi_{W}(\sigma )
    # samples drawn from pi(.; params_j)
    params_j                : Any
    configs_j               : Array         # (n_j, ...)
    
    # optional, kept for symmetry
    probabilities_j         : Optional[Array] = None
    # penalty parameter
    beta_j                  : float = 0.0
    # The final rations r_le (lower to excited) [configs_excited] and r_el (excited to lower) [configs]
    r_le                    : np.array = None  # (n_j,) complex; ratios Psi_{W}(\sigma )/Psi_{W_j}(\sigma ) on configs_j
    r_el                    : np.array = None  # (n_j,) complex; ratios E_loc^{W}(\sigma )Psi_{W}(\sigma )/Psi_{W_j}(\sigma ) on configs
    # backend
    backend_np              : Any = jnp if JAX_AVAILABLE else np
    
    def compute_ratios(self):
        '''
        Compute the ratios needed for the penalty terms.
        The results are stored in self.r_le and self.r_el.
        '''
        
        if self.excited_on_lower.shape != self.lower_on_lower.shape:
            raise ValueError(f"Shapes of excited_on_lower {self.excited_on_lower.shape} and lower_on_lower {self.lower_on_lower.shape} do not match.")
        
        if self.excited_on_excited.shape != self.lower_on_excited.shape:
            raise ValueError(f"Shapes of excited_on_excited {self.excited_on_excited.shape} and lower_on_excited {self.lower_on_excited.shape} do not match.")
        
        if self.excited_on_lower.shape != self.excited_on_excited.shape:
            raise ValueError(f"Shapes of excited_on_lower {self.excited_on_lower.shape} and excited_on_excited {self.excited_on_excited.shape} do not match.")
        
        #! compute r_le
        self.r_le = self.backend_np.exp(self.lower_on_excited - self.excited_on_excited)
        
        #! compute r_el
        self.r_el = self.backend_np.exp(self.excited_on_lower - self.lower_on_lower)

    def __post_init__(self):
        self.r_le = jnp.zeros_like(self.configs_j, dtype=jnp.complex64)
        self.r_el = jnp.zeros_like(self.configs_j, dtype=jnp.complex64)
        self.compute_ratios()

#################################################################

class TDVP:
    r'''
    This class implements the Time-Dependent Variational Principle (TDVP) for quantum systems.
    
    In principle, it is used to calculate the time evolution (or imaginary time evolution) of a quantum state.
    The TDVP is a variational method that uses the time-dependent Schrödinger equation to evolve a quantum state in time.
    
    It is used to be able to calculate:
    
    a) The force vector (can be treated already as a gradient in the first order)
        :math:`F_k=\\langle \\mathcal O_{\\theta_k}^* E_{loc}^{\\theta}\\rangle_c`
        
        i) Calculation of the force vector:
            :math:`\\mathcal O_{\\theta_k}^*` is logarithmic derivative of the wavefunction
        ii) Calculation of the local energy:
            :math:`E_{loc}^{\\theta}` is the local energy of the wavefunction
        iii) Calculation of the connected correlation function: 
            :math:`\\langle \\cdots \\rangle_c` is the connected correlation function
        iv*) The force vector can be appended with penalty terms [excited state search]:
            :math:`F_k=\\langle \\mathcal O_{\\theta_k}^* E_{loc}^{\\theta}\\rangle_c + \\sum_j \\beta_j
            \\langle (\\frac{\\psi _{W_j}}{\\psi _{W}}) - \\langle \\frac{\\psi _{W_j}}{\\psi _{W}} \\rangle ) O_k^\dag \\rangle_c`
            \\langle \\frac{\\psi _{W}}{\\psi _{W_j}} \\rangle`
    b) The quantum Fisher matrix [covariance matrix]
        :math:`S_{k,k'} = \\langle (\\mathcal O_{\\theta_k})^* \\mathcal O_{\\theta_{k'}}\\rangle_c`
    
    and for real parameters :math:`\\theta\\in\\mathbb R`, the TDVP equation reads
        :math:`q\\big[S_{k,k'}\\big]\\dot\\theta_{k'} = -q\\big[xF_k\\big]`
        
    Here, either :math:`q=\\text{Re}` or :math:`q=\\text{Im}` and :math:`x=1` for ground state
    search or :math:`x=i` (the imaginary unit) for real time dynamics.
    
    For ground state search a regularization controlled by a parameter :math:`\\rho` can be included
    by increasing the diagonal entries and solving

        :math:`q\\big[(1+\\rho\\delta_{k,k'})S_{k,k'}\\big]\\theta_{k'} = -q\\big[F_k\\big]`

    The `TDVP` class solves the TDVP equation by computing a pseudo-inverse of :math:`S` via
    eigendecomposition yielding

        :math:`S = V\\Sigma V^\\dagger`

    with a diagonal matrix :math:`\\Sigma_{kk}=\\sigma_k`. In order to do this, it uses the backend specified
    (either JAX or NumPy) and a provided ODE solver.
    
    Assuming that :math:`\\sigma_1` is the smallest eigenvalue, the pseudo-inverse is constructed 
    from the regularized inverted eigenvalues

        :math:`\\tilde\\sigma_k^{-1}=\\frac{1}{\\Big(1+\\big(\\frac{\\epsilon_{SVD}}{\\sigma_j/\\sigma_1}\\big)^6\\Big)\\Big(1+\\big(\\frac{\\epsilon_{SNR}}{\\text{SNR}(\\rho_k)}\\big)^6\\Big)}`

    with :math:`\\text{SNR}(\\rho_k)` the signal-to-noise ratio of :math:`\\rho_k=V_{k,k'}^{\\dagger}F_{k'}` (see `[arXiv:1912.08828] <https://arxiv.org/pdf/1912.08828.pdf>`_ for details).
    
    '''
    
    @dataclass
    class TDVPCompiledMeta:
        gradient_fn         : Callable      # Gradient calculator for a given NN        (not jitted)
        loss_c_fn           : Callable      # Centered loss function calculator         (not jitted)
        deriv_c_fn          : Callable      # Centered derivative function calculator   (not jitted)
        covariance_fn       : Callable      # Covariance matrix calculator              (not jitted)
        prepare_fn          : Callable      # Preparation function
        prepare_fn_m        : Callable      # Preparation function for the excited state
        # jitted versions
        gradient_fn_j       : Callable
        loss_c_fn_j         : Callable
        deriv_c_fn_j        : Callable
        covariance_fn_j     : Callable
        prepare_fn_j        : Callable
        prepare_fn_m_j      : Callable
        # flags
        is_jax              : bool
    
    def __init__(
            self,
            use_sr          : bool                                      = True,
            use_minsr       : bool                                      = False,
            rhs_prefactor   : Union[float, complex]                     = 1.0,
            *,
            # other
            regularization  : float                                     = 0.0,      # maybe used?
            # Stochastic reconfiguration parameters
            sr_lin_solver   : Optional[Union[solvers.Solver, str]]      = None,     # default solver
            sr_precond      : Optional[precond.Preconditioner]          = None,     # default preconditioner
            sr_snr_tol      : float                                     = 1e-3,     # for signal-to-noise ratio
            sr_pinv_tol     : float                                     = 1e-14,    # for Moore-Penrose pseudo-inverse - tolerance of eigenvalues
            sr_pinv_cutoff  : float                                     = 1e-8,     # for Moore-Penrose pseudo-inverse - cutoff of eigenvalues
            sr_diag_shift   : float                                     = 0.0,      # diagonal shift for the covariance matrix in case of ill-conditioning
            sr_lin_solver_t : Optional[Union[solvers.SolverForm, str]]  = 'gram',   # form of the solver - gram, matrix
            sr_lin_x0       : Optional[Array]                           = None,     # initial guess for the solver
            sr_maxiter      : int                                       = 100,      # maximum number of iterations for the linear solver
            # Backend
            backend         : str                                       = 'default',# 'jax' or 'numpy'
            logger          : Optional[Any]                             = None
        ):
        ''' TDVP Initialization Function
        
        This function initializes the TDVP class with the provided parameters. It sets up the necessary configurations for performing
        time-dependent variational principle calculations on quantum states. The parameters allow customization of the stochastic reconfiguration method, solver options, and backend settings.
        Parameters
        ----------
        use_sr : bool, optional
            Flag to indicate if stochastic reconfiguration is used, by default True.
        use_minsr : bool, optional
            Flag to indicate if minimal stochastic reconfiguration is used, by default False. This reverses the order of operators in the covariance matrix.
            We do this when the number of parameters is large compared to the number of samples (usually when the ansatz is very expressive).
            Thus, instead of computing S = <O^+ O>_c, we compute S = <O O^+>_c which is smaller in size...
        rhs_prefactor : Union[float, complex], optional
            Prefactor for the RHS of the TDVP equation (-1 for ground state, -i for real time), by default 1.0.
        sr_lin_solver : Optional[solvers.Solver], optional
            Linear solver for the TDVP equation (A x = b), by default None (uses default solver). One can
            provide a solver instance or a string identifier:
            
            Currently, it is recomended to use one of the following solvers:
                - PseudoInverseSolver           ('pseudo_inverse') - ONLY FOR FULL CORRELATION MATRIX FORMING
                - ConjugateGradientSolver       ('cg' - inner implementation), ('scipy_cg' - scipy implementation) - USES GRAM MATRIX STRUCTURE
                - MinResSolver                  ('minres' - inner implementation), ('scipy_minres' - scipy implementation) - USES GRAM MATRIX STRUCTURE
                - MinresQLPSolver               ('minres_qlp' - inner implementation) - USES GRAM MATRIX STRUCTURE
                - BackendSolver                 ('backend' - uses default backend solver, e.g., numpy.linalg.solve)
        sr_precond : Optional[precond.Preconditioner], optional
            Preconditioner for the TDVP equation (A x = b), by default None (uses default preconditioner). One can
            provide a preconditioner instance or a string identifier. 
            
            Currently, it is recomended to use one of the following preconditioners:
            TODO:
        sr_snr_tol : float, optional
            TODO: Start to use this
            Tolerance for the signal-to-noise ratio in the pseudo-inverse calculation, by default 1e-3.
        sr_pinv_tol : float, optional
            Tolerance for the Moore-Penrose pseudo-inverse eigenvalues, by default 1e-14.
        sr_pinv_cutoff : float, optional
            Cutoff for the Moore-Penrose pseudo-inverse eigenvalues, by default 1e-8.
        sr_diag_shift : float, optional
            Diagonal shift for the covariance matrix in case of ill-conditioning, by default 0.0.
            Can be used to stabilize the inversion of the covariance matrix -> A -> A + sr_diag_shift * I
        sr_lin_solver_t : Optional[solvers.SolverForm], optional
            Form of the solver - gram, matrix, by default solvers.SolverForm.GRAM.
            Determines whether to use the Gram matrix structure or form the full matrix. Options are:
                - solvers.SolverForm.GRAM   : Uses the Gram matrix structure (S = <O^+ O>_c)
                - solvers.SolverForm.MATVEC : Uses matrix-vector products
                - solvers.SolverForm.FULL   : Forms the full covariance matrix
            Or can be provided as a string: 'gram', 'matvec', 'full'
            
        sr_lin_x0 : Optional[Array], optional
            Initial guess for the solver, by default None. Can start from zero or previous solution. Only used for iterative solvers.
        sr_maxiter : int, optional
            Maximum number of iterations for the linear solver, by default 100. Only used for iterative solvers.
        backend : str, optional
            Backend for numerical operations - 'jax' or 'numpy', by default 'default' (uses JAX if available, otherwise NumPy).        
        '''
        
        self.backend         = get_backend(backend)
        self.is_jax          = not (backend == 'numpy' or backend == 'np' or backend == np)
        self.is_np           = not self.is_jax
        self.backend_str     = 'jax' if self.is_jax else 'numpy'
        
        self.use_sr          = use_sr               # flag to indicate if stochastic reconfiguration is used
        self.use_minsr       = use_minsr            # flag to indicate if minimal SR is used -> reverse O^+O to O O^+
        self.rhs_prefactor   = rhs_prefactor        # prefactor for the RHS of the TDVP equation (1 for ground state, i for real time)
        self.form_matrix     = False                # flag to indicate if the full matrix is formed
        self.regularization  = regularization       # maybe used?
        
        #! handle the stochastic reconfiguration parameters
        self.sr_snr_tol      = sr_snr_tol           # for signal-to-noise ratio
        self.sr_pinv_tol     = sr_pinv_tol          # for Moore-Penrose pseudo-inverse - tolerance of eigenvalues
        self.sr_pinv_cutoff  = sr_pinv_cutoff       # for Moore-Penrose pseudo-inverse - cutoff of eigenvalues
        self.sr_diag_shift   = sr_diag_shift        # diagonal shift for the covariance matrix in case of ill-conditioning
        self.sr_maxiter      = sr_maxiter           # maximum number of iterations for the linear solver

        #! handle the solver
        try:
            self.sr_solve_lin    = None             # linear solver for the TDVP equation (A x = b)
            self.sr_solve_lin_t  = sr_lin_solver_t  # form of the solver
            self.sr_solve_lin_fn = None             # function handle for the solver
            self.set_solver(sr_lin_solver)
        except Exception as e:
            warnings.warn(f"Solver could not be set: {e}")
            raise e
            
        #! handle the preconditioner
        try:
            self.sr_precond      = None             # preconditioner for the TDVP equation (A x = b)
            self.sr_precond_fn   = None             # function handle for the preconditioner
            self.set_preconditioner(sr_precond)
        except Exception as e:
            warnings.warn(f"Preconditioner could not be set: {e}")
            raise e

        self.meta           = None
        
        # Helper storage
        self._e_local_mean  = None     # mean local energy
        self._e_local_std   = None     # std local energy
        # ---
        self._solution      = None     # solution of the TDVP equation
        self._f0            = None     # force vector obtained from the covariance of loss and derivative
        self._s0            = None     # Fisher matrix obtained from the covariance of derivatives
        self._n_samples     = None     # number of samples
        self._full_size     = None     # full size of the covariance matrix
        self._x0            = sr_lin_x0
        self.timings        = TDVPTimes()
        
        # Global phase parameter tracking for dynamical correlation functions
        # Equation from paper: $|\psi_{\theta_0,\theta}\rangle = e^{\theta_0}|\psi_\theta\rangle$
        # Evolution: $\dot{\theta}_0 = -i\langle\hat{H}\rangle - \dot{\theta}_k\langle\psi_\theta|\partial_{\theta_k} \psi_\theta\rangle$
        self._theta0        = 0.0      # global phase parameter
        self._theta0_dot    = 0.0      # time derivative of global phase parameter

        # Logger
        self.logger         = logger
        
        #! functions
        self._init_functions()

    ###################
    #! TIMING
    ###################

    @contextmanager
    def _time(self, phase: str, fn, *args, **kwargs):
        """
        Context manager to time a function call and store elapsed time.

        Yields:
            result of fn(*args, **kwargs)
        """
        result, elapsed = timeit(fn, *args, **kwargs)
        self.timings[phase] = elapsed
        yield result

    ###################
    #! INITIALIZATION
    ###################
    
    def _init_functions(self):
        """
        Initializes and assigns function handles for gradient, loss, derivatives, covariance, and preparation routines
        based on the current backend (JAX or NumPy) and configuration flags.
        - Selects appropriate functions from the `sr` module depending on whether JAX or NumPy is used (`self.is_jax`).
        - Chooses between standard and minimal SR covariance functions based on `self.use_minsr`.
        - Assigns both standard and modified preparation functions.
        - If using JAX, applies `jax.jit` to the selected functions for just-in-time compilation and stores them with `_j` suffix.
        - If not using JAX, stores the original function references with `_j` suffix for interface consistency.
        This method should be called during initialization to ensure all computational routines are set up according to the
        current backend and configuration.
        """
        
        self._gradient_fn           = sr.gradient_jax               if self.is_jax else sr.gradient_np
        self._loss_c_fn             = sr.loss_centered_jax          if self.is_jax else sr.loss_centered_np
        self._deriv_c_fn            = sr.derivatives_centered_jax   if self.is_jax else sr.derivatives_centered_np
        
        # covariance functions - standard and minsr
        if self.use_minsr:
            self._covariance_fn     = sr.covariance_jax_minsr       if self.is_jax else sr.covariance_np_minsr
        else:
            self._covariance_fn     = sr.covariance_jax             if self.is_jax else sr.covariance_np

        # modified and standard preparation functions
        self._prepare_fn            = sr.solve_jax_prepare                  if self.is_jax else sr.solve_numpy_prepare
        self._prepare_fn_m          = sr.solve_jax_prepare_modified_ratios  if self.is_jax else sr.solve_numpy_prepare
            
        try:
            #! store the jitted functions
            if self.is_jax:
                self._gradient_fn_j     = self._gradient_fn
                self._loss_c_fn_j       = self._loss_c_fn
                self._deriv_c_fn_j      = self._deriv_c_fn
                self._covariance_fn_j   = self._covariance_fn
                self._prepare_fn_j      = self._prepare_fn
                self._prepare_fn_m_j    = self._prepare_fn_m
            else:
                self._gradient_fn_j     = self._gradient_fn
                self._loss_c_fn_j       = self._loss_c_fn
                self._deriv_c_fn_j      = self._deriv_c_fn
                self._covariance_fn_j   = self._covariance_fn
                self._prepare_fn_j      = self._prepare_fn
                self._prepare_fn_m_j    = self._prepare_fn_m
        except Exception as e:
            raise e

    def _init_solver_lin(self):
        r"""
        Initializes the linear solver for the stochastic reconfiguration (SR) method.
        This method sets up the appropriate solver function (`sr_solve_lin_fn`) based on the type of solver specified
        by `sr_solve_lin_t` and the solver instance `sr_solve_lin`. It determines whether to form the full matrix or use
        matrix-vector products, and configures the solver function with the correct backend and options.
        Raises:
            ValueError: If SR is enabled (`use_sr` is True) but no solver (`sr_solve_lin`) is set.
        Attributes Set:
            self.form_matrix (bool):
                Indicates whether to form the full matrix or use matrix-vector products.
            self.sr_solve_lin_fn (callable):
                The configured solver function for SR linear systems.
        """
        
        if self.sr_solve_lin is None:
            # Default to PseudoInverseSolver if no solver is provided
            self.sr_solve_lin_t     = solvers.SolverForm.MATRIX
            self.sr_solve_lin       = solvers.PseudoInverseSolver(backend=self.backend, sigma=self.sr_diag_shift)
            if self.logger: self.logger.info("No solver provided. Defaulting to PseudoInverseSolver with full matrix formation.")

        if isinstance(self.sr_solve_lin, str) or isinstance(self.sr_solve_lin, solvers.SolverType) or isinstance(self.sr_solve_lin, int):
            identifier              = self.sr_solve_lin
            self.sr_solve_lin       = solvers.choose_solver(
                                        solver_id       =   identifier, 
                                        is_gram         =   self.sr_solve_lin_t == solvers.SolverForm.GRAM.value,
                                        sigma           =   self.sr_diag_shift, 
                                        backend         =   self.backend, 
                                        default_precond =   self.sr_precond, 
                                        maxiter         =   self.sr_maxiter)
            if self.logger: self.logger.info(f"Solver set to {self.sr_solve_lin} based on identifier '{identifier}'.")

        if self.sr_solve_lin is not None:
            if self.sr_solve_lin_t == solvers.SolverForm.GRAM.value:
                self.form_matrix    = False
            elif self.sr_solve_lin_t == solvers.SolverForm.MATVEC.value:
                self.form_matrix    = False
            else:
                self.form_matrix    = True
                
            #! set the solver function
            self.sr_solve_lin_fn = self.sr_solve_lin.get_solver_func(
                    backend_module  = self.backend,
                    use_matvec      = self.sr_solve_lin_t == solvers.SolverForm.MATVEC.value,
                    use_fisher      = self.sr_solve_lin_t == solvers.SolverForm.GRAM.value,
                    use_matrix      = self.form_matrix,
                    sigma           = self.sr_diag_shift,
                )
            
        elif self.use_sr and self.sr_solve_lin is None:
            raise ValueError('The solver is not set. Please set the solver or use the default one.') 

    def _init_preconditioner(self):
        """
        Initializes the preconditioner function for the stochastic reconfiguration (SR) solver
        based on the selected solver form.

        Depending on the value of `self.sr_solve_lin_t`, this method assigns the appropriate
        preconditioner application function from `self.sr_precond` to `self.sr_precond_fn`:
            - If `GRAM`,    uses `get_apply_gram()`
            - If `MATVEC`,  uses `get_apply()`
            - If `MATRIX`,  uses `get_apply_mat()`
        Raises a ValueError if the solver form is unrecognized or if the preconditioner is not set.

        Raises:
            ValueError: If the preconditioner is not set or the solver form is invalid.
        """
        if  isinstance(self.sr_precond, str) or                                 \
            isinstance(self.sr_precond, precond.PreconditionersTypeSym) or      \
            isinstance(self.sr_precond, precond.PreconditionersTypeNoSym) or    \
            isinstance(self.sr_precond, precond.PreconditionersType) or         \
            hasattr(self.sr_precond, 'value') and isinstance(self.sr_precond.value, int):
            
            self.sr_precond = precond.choose_precond(precond_id=self.sr_precond, backend=self.backend)
        
        if self.sr_precond is not None:
            if self.sr_solve_lin_t == solvers.SolverForm.GRAM.value:
                self.sr_precond_fn = self.sr_precond.get_apply_gram()
            elif self.sr_solve_lin_t == solvers.SolverForm.MATVEC.value:
                self.sr_precond_fn = self.sr_precond.get_apply()
            elif self.sr_solve_lin_t == solvers.SolverForm.MATRIX.value:
                self.sr_precond_fn = self.sr_precond.get_apply_mat()
            else:
                raise ValueError('The preconditioner is not set. Please set the preconditioner or use the default one.')  
    
    ###################
    #! SETTERS
    ###################
    
    def set_solver(self, solver: Callable):
        '''
        Set the solver for the TDVP equation.
        
        Parameters
        ----------
        solver : Callable
            The solver function to be used for the TDVP equation.
        '''
        self.sr_solve_lin = solver
        self._init_solver_lin()
    
    def set_preconditioner(self, precond: Callable):
        '''
        Set the preconditioner for the TDVP equation.
        
        Parameters
        ----------
        precond : Callable
            The preconditioner function to be used for the TDVP equation.
        '''
        self.sr_precond = precond
        self._init_preconditioner()
    
    def set_useminsr(self, use_minsr: bool):
        '''
        Set the use of minimum stochastic reconfiguration (minsr) method.
        
        Parameters
        ----------
        use_minsr : bool
            Whether to use the minimum stochastic reconfiguration (minsr) method.
        '''
        self.use_minsr = use_minsr
        if self.use_minsr:
            self._covariance_fn = sr.covariance_jax_minsr if self.is_jax else sr.covariance_np_minsr
        else:
            self._covariance_fn = sr.covariance_jax if self.is_jax else sr.covariance_np

    def set_diag_shift(self, diag_shift: float):
        '''
        Set the diagonal shift for the TDVP equation.
        
        The diagonal shift (sigma) is now passed dynamically at solve time,
        so changing this value does NOT trigger solver recompilation.
        
        Parameters
        ----------
        diag_shift : float
            The diagonal shift to be used for the TDVP equation.
        '''
        self.sr_diag_shift = diag_shift

    def set_regularization(self, regularization: float):
        '''
        Set the regularization parameter for the TDVP equation.
        
        Parameters
        ----------
        regularization : float
            The regularization parameter to be used for the TDVP equation.
        '''
        self.regularization = regularization
        
    ###################
    
    def set_rhs_prefact(self, rhs_prefactor: Union[float, complex]):
        '''
        Set the right-hand side prefactor for the TDVP equation.
        This is used to scale the right-hand side of the TDVP equation.
        Parameters
        ----------
        rhs_prefactor : Union[float, complex]
            The right-hand side prefactor to be used for the TDVP equation.
        '''
        self.rhs_prefactor = rhs_prefactor
    
    def set_snr_tol(self, snr_tol: float):
        '''
        Set the signal-to-noise ratio tolerance for the TDVP equation.
        
        Parameters
        ----------
        snr_tol : float
            The signal-to-noise ratio tolerance to be used for the TDVP equation.
        '''
        self.sr_snr_tol = snr_tol
    
    def set_pinv_tol(self, pinv_tol: float):
        '''
        Set the pseudo-inverse tolerance for the TDVP equation.
        This is used to determine the cutoff for the pseudo-inverse calculation.
        Parameters
        ----------
        pinv_tol : float
            The pseudo-inverse tolerance to be used for the TDVP equation.
        '''
        self.sr_pinv_tol = pinv_tol
    
    ###################
    #! GETTERS
    ###################
    
    def get_loss_centered(self, loss, loss_m = None):
        '''
        Get the centered loss:
        
        Calculates <E_loc>_c = E_loc - <E_loc>
        
        Parameters
        ----------
        loss : Array
            The loss to be centered.
        loss_m : Optional[Array]
            The mean loss to be used for centering.
        
        Returns
        -------
        Array
            The centered loss.
        '''
        return self._loss_c_fn_j(loss, loss.mean(axis = 0) if not loss_m else loss_m)
    
    def get_deriv_centered(self, deriv, deriv_m = None):
        '''
        Get the centered derivative.
        
        Calculates <O_k>_c = O_k - <O_k>
        
        Parameters
        ----------
        deriv : Array
            The derivative to be centered.
        
        Returns
        -------
        Array
            The centered derivative.
        '''
        deriv_m = deriv.mean(axis = 0) if not deriv_m else deriv_m
        return self._deriv_c_fn_j(deriv, deriv_m)
    
    ##################
    #! TDVP
    ##################
    
    def _get_tdvp_standard_inner(self, loss, log_deriv, **kwargs):
        '''
        Get the standard TDVP loss and derivative.
        
        Parameters
        ----------
        loss : Array
            The loss to be used for the TDVP equation.
        log_deriv : Array
            The logarithm of the derivative to be used for the TDVP equation.
        betas : Optional[Array]
            The betas to be used for the excited states.
        r_psi_low_ov_exc : Optional[Array]
            The ratios of the wavefunction for the low-lying excited states.
        r_psi_exc_ov_low : Optional[Array]
            The ratios of the wavefunction for the excited states.
        Returns
        -------
        Tuple[Array, Array]
            The standard TDVP loss and derivative.
        '''
        
        # optional parameters for excited states
        excited_penalty: List[TDVPLowerPenalty] = kwargs.get('lower_states', None)
        
        #! centered loss and derivative
        if excited_penalty is None or len(excited_penalty) == 0:
            (loss_c, var_deriv_c, var_deriv_c_h, self._n_samples, self._full_size) = self._prepare_fn_j(loss, log_deriv)
        else:
            betas           = self.backend.array([x.beta_j for x in excited_penalty])
            r_el            = self.backend.array([x.r_el for x in excited_penalty])
            r_le            = self.backend.array([x.r_le for x in excited_penalty])
            (loss_c, var_deriv_c, var_deriv_c_h, self._n_samples, self._full_size) = self._prepare_fn_m_j(loss, log_deriv, betas, r_el, r_le)        
        return loss_c, var_deriv_c, var_deriv_c_h
    
    def get_tdvp_standard(self, loss, log_deriv, **kwargs):
        '''
        Get the standard TDVP loss and derivative.
        
        Parameters
        ----------
        loss : Array
            The loss to be used for the TDVP equation.
        log_deriv : Array
            The logarithm of the derivative to be used for the TDVP equation.
        minsr : Optional[bool]
            Whether to use the minimum stochastic reconfiguration (minsr) method.
        Returns
        -------
        Tuple[Array, Array]
            The standard TDVP loss and derivative.
        '''
        
        #! state information
        self._e_local_mean  = self.backend.mean(loss, axis=0)
        self._e_local_std   = self.backend.std(loss, axis=0)

        with self._time('prepare', self._get_tdvp_standard_inner, loss, log_deriv, **kwargs) as prepare:
            loss_c, var_deriv_c, var_deriv_c_h = prepare
            
        # for minsr, it is unnecessary to calculate the force vector, however, do it anyway for now
        with self._time('gradient', self._gradient_fn_j, var_deriv_c_h, loss_c, self._n_samples) as gradient:
            self._f0 = gradient
        self._s0 = None
        
        # MinSR uses T (N_s x N_s), calculating S (N_p x N_p) is wasteful and causes OOM
        if self.form_matrix and not self.use_minsr: 
            with self._time('covariance', self._covariance_fn_j, var_deriv_c, var_deriv_c_h, self._n_samples) as covariance:
                self._s0 = covariance
        
        return self._f0, self._s0, (loss_c, var_deriv_c, var_deriv_c_h)

    ##################
    #! SOLVERS
    ##################
    
    def _solve_prepare_matvec(self, mat_s: Array, mat_sp: Array):
        """
        Prepares the covariance matrix and loss vector for the linear system to be solved.
        This method is used to prepare the input data for the linear solver, including
        centering the covariance matrix and loss vector.
        Parameters
        ----------
        mat_s : Array
            The covariance matrix (centered). [N_samples, N_variational]
        mat_sp : Array
            The covariance matrix (centered - complex conjugate). [N_variational, N_samples]
        Returns
        -------
        Tuple[Array, Array]
            The prepared covariance matrix and loss vector.
        """
        def _matvec(v, sigma):
            inter = self.backend.matmul(mat_s, v)
            return self.backend.matmul(mat_sp, inter) / v.shape[0] + sigma * v
        return jax.jit(_matvec) if self.is_jax else _matvec
    
    def _solve_prepare_s_and_loss(self, vd_c: Array, vd_c_h: Array, loss_c: Array, forces: Array):
        """
        Prepares the covariance matrix and loss vector for the linear system to be solved.
        This method is used to prepare the input data for the linear solver, including
        centering the covariance matrix and loss vector.
        Parameters
        ----------
        vd_c : Array
            The covariance matrix (centered). [N_samples, N_variational]
        vd_c_h : Array
            The covariance matrix (centered - complex conjugate). [N_variational, N_samples]
        loss_c : Array
            The loss vector (centered). [N_samples,]
        forces : Array
            The forces vector (centered). [N_variational,]
        Returns
        -------
        Tuple[Array, Array]
            The prepared covariance matrix and loss vector.
        """
        if self.use_minsr:
            # T = O @ O^dag
            # Solver expects (s, s_p) -> s @ s_p
            # Scaling: Equation is T * x = E/N. The RHS passed here should be centered loss E_c.
            # The scaling by 1/N is usually handled inside the solver or covariance func. 
            # In sr.solve_jax, we handle scaling. Here we just prepare raw inputs.
            # return (vd_c, vd_c_h), loss_c
            return (vd_c_h, vd_c), loss_c
        
        # otherwise, standard TDVP equation is solved
        # return (vd_c_h, vd_c), forces
        return (vd_c, vd_c_h), forces

    def _solve_choice(  self, 
                        vec_b           : Array,
                        solve_func      : Callable,
                        mat_s           : Optional[Array] = None,
                        mat_s_p         : Optional[Array] = None,
                        mat_a           : Optional[Array] = None,
                    ) -> solvers.SolverResult:
        """
        Solves a linear system using a specified solver form and configuration.
        Depending on the value of `self.sr_solve_lin_t`, this method dispatches to the appropriate
        solver function with the required arguments. Supports different solver forms such as GRAM and MATRIX,
        and handles special cases like the minimum stochastic reconfiguration (minsr) method.
        Parameters
        ----------
        mat_s : Optional[Array]
            The first matrix to be used in the linear system. Required for GRAM and MATVEC solver forms.
        mat_s_p : Optional[Array]
            The second matrix to be used in the linear system. Required for GRAM and MATVEC solver forms.
        mat_a : Optional[Array]
            The matrix to be used in the linear system. Required for MATRIX solver form.
        vec_b : Array
            The right-hand side vector of the linear system.
        solve_func : Callable
            The function to be used for solving the linear system.
        Returns
        -------
        solution : Array or None
            The solution to the linear system, or None if the solver form is not recognized.
        Raises
        ------
        ValueError
            If the matrix solver is selected with the minimum stochastic reconfiguration (minsr) method,
            which is not implemented.
        """
        
        solution = None
        if self.sr_solve_lin_t == solvers.SolverForm.GRAM.value:
            solution = solve_func(s             =   mat_s,
                                s_p             =   mat_s_p,
                                b               =   vec_b,
                                x0              =   self._x0,
                                precond_apply   =   self.sr_precond_fn,
                                maxiter         =   self.sr_maxiter,
                                tol             =   self.sr_pinv_tol,
                                sigma           =   self.sr_diag_shift)
        elif self.sr_solve_lin_t == solvers.SolverForm.MATRIX.value:
            if self.use_minsr:
                raise ValueError('The matrix solver is not implemented for the minimum stochastic reconfiguration (minsr) method.')
            solution = solve_func(a             =   mat_a,
                                b               =   vec_b,
                                x0              =   self._x0,
                                precond_apply   =   self.sr_precond_fn,
                                maxiter         =   self.sr_maxiter,
                                tol             =   self.sr_pinv_tol,
                                sigma           =   self.sr_diag_shift)  # Dynamic sigma!
            
        elif self.sr_solve_lin_t == solvers.SolverForm.MATVEC.value:
            # For MATVEC mode, create a matvec with current sigma bound
            matvec_fn                           = self._solve_prepare_matvec(mat_s, mat_s_p)
            matvec_with_sigma                   = lambda v, mv=matvec_fn, sig=self.sr_diag_shift: mv(v, sig)
            solution                            = solve_func(matvec         =   matvec_with_sigma,
                                                            b               =   vec_b,
                                                            x0              =   self._x0,
                                                            precond_apply   =   self.sr_precond_fn,
                                                            maxiter         =   self.sr_maxiter,
                                                            tol             =   self.sr_pinv_tol)
        return solution
    
    def _solve_handle_x0(self, vec_b: Array, use_old_result: bool):
        """
        Handles the initial guess for the linear solver.
        If `use_old_result` is True, it uses the previous solution as the initial guess.
        Otherwise, it initializes the guess to zero.
        Parameters
        ----------
        vec_b : Array
            The right-hand side vector of the linear system.
        use_old_result : bool
            Whether to use the previous solution as the initial guess.
        Returns
        -------
        Array
            The initial guess for the linear solver.
        """
        if use_old_result:
            self._x0 = self._solution
        if self._x0 is None or self._x0.shape != vec_b.shape:
            self._x0 = self.backend.zeros_like(vec_b)
        return self._x0

    ###############
    #! GLOBAL PHASE
    ###############

    def compute_global_phase_evolution(self, mean_energy: Array, param_derivatives: Array, log_derivatives: Array):
        r"""
        Compute the evolution of the global phase parameter $\theta_0$.
        
        Based on Equation (5) from PHYSICAL REVIEW LETTERS 131, 046501 (2023):
        
        $\dot{\theta}_0 = -i\langle\hat{H}\rangle - \dot{\theta}_k\langle\psi_\theta|\partial_{\theta_k} \psi_\theta\rangle$
        
        The global phase parameter tracks the overall phase of the wavefunction,
        which is important for computing dynamical correlation functions.
        
        Parameters
        ----------
        mean_energy : Array
            The mean energy $\langle\hat{H}\rangle$ from local energy computation
        param_derivatives : Array
            The time derivatives $\dot{\theta}_k$ from solving the TDVP equation
        log_derivatives : Array
            The centered log derivatives $\langle\psi_\theta|\partial_{\theta_k} \psi_\theta\rangle$ from the covariance computation
        
        Returns
        -------
        Array
            The computed time derivative $\dot{\theta}_0$ for the global phase parameter
        """
        # First term: -i * <H> -> average energy
        term1       = -1j * mean_energy
        
        # Second term: - <dot{theta}_k * <psi|d_theta_k psi>>
        # This is the contraction of parameter derivatives with log derivatives
        term2       = -self.backend.sum(param_derivatives * log_derivatives)
        
        theta0_dot  = term1 + term2
        # self._theta0_dot = theta0_dot
        return theta0_dot

    ###################
    #! MAIN SOLVE
    ###################

    def solve(self, e_loc: Array, log_deriv: Array, **kwargs):
        #? Get the lower states information penalty
        
        # obtain the loss and covariance without the preprocessor
        self._f0, self._s0, (tdvp)  = self.get_tdvp_standard(e_loc, log_deriv, **kwargs)
        
        f = self._f0                # the force vector
        s = self._s0                # the covariance matrix, if formed
        
        #! handle the solver
        solve_func                  = self.sr_solve_lin_fn
        loss_c, vd_c, vd_c_h        = tdvp

        #! handle the preprocessor
        (mat_s, mat_s_p), vec_b     = self._solve_prepare_s_and_loss(vd_c, vd_c_h, loss_c, f)
        
        #! handle the initial guess - use the previous solution
        with self._time('x0', self._solve_handle_x0, vec_b, kwargs.get('use_old_result', False)) as x0:
            self._x0 = x0

        #! prepare the rhs
        if self.rhs_prefactor != 1.0:
            vec_b = vec_b * self.rhs_prefactor

        #! if not using SR, return the negative force vector as the solution
        if not self.use_sr:
            return solvers.SolverResult(
                x               = -vec_b,
                iterations      = 0,
                residual_norm   = 0.0,
                converged       = True,
            )

        #! precondition the S matrices, use regularization?
        if True:
            #!TODO: Do something with regularization, sr_ and others...
            pass
    
        #! solve the linear system
        with self._time('solve', self._solve_choice, vec_b=vec_b, mat_s=mat_s, mat_s_p=mat_s_p, mat_a=s, solve_func=solve_func) as solve:
            solution = solve
        
        if self.use_minsr and solution is not None:
            new_solution = solvers.SolverResult(
                x               = self.backend.matmul(vd_c_h, solution.x),
                iterations      = solution.iterations,
                residual_norm   = solution.residual_norm,
                converged       = solution.converged,
            )
            solution = new_solution
        
        # Compute global phase evolution $\dot{\theta}_0$
        # $\dot{\theta}_0 = -i\langle\hat{H}\rangle - \dot{\theta}_k\langle\psi_\theta|\partial_{\theta_k} \psi_\theta\rangle$
        if solution is not None and self.rhs_prefactor:
            param_derivatives           = solution.x if hasattr(solution, 'x') else solution
            log_derivatives_centered    = vd_c # centered log derivatives
            self._theta0_dot            = self.compute_global_phase_evolution(self._e_local_mean, param_derivatives, log_derivatives_centered)
            
        #! save the solution
        self._solution = solution
        return solution
    
    #########################
    #! Representation
    #########################    
    
    def __call__(self, net_params, t, *, est_fn, configs, configs_ansatze, probabilities, **kwargs):
        '''
        Call the TDVP class to compute the time evolution of the quantum state.
        
        Parameters
        ----------
        net_params : Array
            The network parameters to be used for the TDVP equation.
        t : float
            The time to be used for the TDVP equation.
        est_fn : Callable
            The function to be used for estimating the TDVP equation.
        configs : Array
            The configurations to be used for the TDVP equation.
        configs_ansatze : Array
            The ansatz configurations to be used for the TDVP equation.
        probabilities : Array
            The probabilities to be used for the TDVP equation.
        
        Returns
        -------
        Array
            The time-evolved quantum state.
        '''
        #! get the loss and derivative in the original scenario 
        #? (MonteCarlo return -> (loss, mean_loss, std_loss), log_deriv, (meta))
        (loss, mean_loss, std_loss), log_deriv, (shapes, sizes, iscpx) = est_fn(net_params, t, configs, configs_ansatze, probabilities)
        
        #! set the meta information
        self.meta       = TDVPStepInfo(
            mean_energy     = mean_loss,
            std_energy      = std_loss,
            failed          = False,
            sr_converged    = False,
            sr_executed     = False,
            sr_iterations   = 0,
            timings         = self.timings
        )
        
        #! obtain the solution

        solution: solvers.SolverResult = self.solve(loss, log_deriv, **kwargs)
                    
        self.meta = TDVPStepInfo(
            mean_energy     = self._e_local_mean,
            std_energy      = self._e_local_std,
            failed          = False,
            sr_converged    = solution.converged,
            sr_executed     = True,
            sr_iterations   = solution.iterations,
            timings         = self.timings,
            theta0_dot      = self._theta0_dot,  # Global phase time derivative
            theta0          = self._theta0       # Current global phase
        )
        
        return solution.x, self.meta, (shapes, sizes, iscpx)
    
    def __repr__(self):
        return f'TDVP(backend={self.backend_str},use_sr={self.use_sr},use_minsr={self.use_minsr},rhs_prefactor={self.rhs_prefactor},sr_snr_tol={self.sr_snr_tol},sr_pinv_tol={self.sr_pinv_tol},sr_diag_shift={self.sr_diag_shift},sr_maxiter={self.sr_maxiter})'
    
    def __str__(self):
        return f'TDVP(backend={self.backend_str},use_sr={self.use_sr},use_minsr={self.use_minsr},rhs_prefactor={self.rhs_prefactor},sr_snr_tol={self.sr_snr_tol},sr_pinv_tol={self.sr_pinv_tol},sr_diag_shift={self.sr_diag_shift},sr_maxiter={self.sr_maxiter})'

    def __len__(self):
        return len(self._solution) if self._solution is not None else 0
    
    def __getitem__(self, key):
        if self._solution is not None:
            return self._solution[key]
        else:
            raise ValueError('The solution is not available. Please run the TDVP solver first.')
    
    #########################
    #! PROPERTIES
    #########################
    
    @property
    def solution(self):
        return self._solution
    
    @property
    def forces(self):
        return self._f0
    
    @property
    def covariance(self):
        return self._s0
    
    @property
    def loss_mean(self):
        return self._e_local_mean
    
    @property
    def loss_std(self):
        return self._e_local_std
    
    @property
    def n_samples(self):
        return self._n_samples
    
    @property
    def full_size(self):
        return self._full_size
    
    #########################
    #! GLOBAL PHASE PROPERTIES
    #########################
    
    @property
    def global_phase(self):
        """Get the current global phase parameter θ₀."""
        return self._theta0
    
    @property
    def global_phase_dot(self):
        """Get the current time derivative of global phase θ̇₀."""
        return self._theta0_dot
    
    def set_global_phase(self, theta0: float):
        """Set the global phase parameter θ₀ (e.g., when loading from checkpoint)."""
        self._theta0 = theta0
    
    def update_global_phase(self, dt: float):
        """
        Integrate the global phase forward in time.
        
        $\\theta_0(t+dt) = \\theta_0(t) + dt \\times \\dot{\\theta}_0(t)$
        
        Parameters
        ----------
        dt : float
            The time step for integration
        """
        self._theta0 += dt * self._theta0_dot
        return self._theta0

#################################################################
#! END OF FILE
#################################################################