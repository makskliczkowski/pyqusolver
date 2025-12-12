"""
Lazy operator loader module.

Provides convenient access to operator modules without requiring explicit imports.
Operators are loaded on-demand when first accessed. This lazy loading mechanism helps reduce initial import times and memory usage.
------------------------------------------------------------------------

Usage:
    # Via Hamiltonian
    ops         = hamil.operators
    sig_x       = ops.sig_x(ns=4, sites=[0, 1])

    # Via HilbertSpace
    ops         = hilbert.operators
    c_dag       = ops.c_dag(ns=4, sites=[0])
    
    # Get matrix representation
    sig_x_mat   = sig_x.matrix
    
Example Operators Available:
- Spin Operators (for SPIN_1_2, SPIN_1):
    sig_x, sig_y, sig_z, sig_p, sig_m, sig_xy, sig_xz, sig_yx, sig_yy, sig_yz, sig_zx, sig_zy
- Fermionic Operators (for SPINLESS_FERMIONS):
    c_dag (creation), c (annihilation), n_op (number operator)

------------------------------------------------------------------------
Author          : Maksymilian Kliczkowski
Date            : November 2025
Email           : maxgrom97@gmail.com
Description     : Lazy loader for operator modules based on LocalSpace type.
------------------------------------------------------------------------
"""

import time
import numpy as np
from typing                                     import Callable, List, Optional, TYPE_CHECKING, Dict

try:
    from QES.Algebra.Hilbert.hilbert_local      import LocalSpaceTypes
    from QES.general_python.common.memory       import get_available_memory_gb, log_memory_status
except ImportError as e:
    raise ImportError("Could not import LocalSpaceTypes. Ensure that QES package is properly installed.") from e

if TYPE_CHECKING:
    from QES.general_python.common.flog         import Logger
    from QES.general_python.lattices.lattice    import Lattice
    from QES.Algebra.Operator                   import Operator
    from QES.Algebra.hilbert                    import HilbertSpace
    from QES.Algebra.Operator                   import operators_spin
    from QES.Algebra.Operator                   import operators_spinless_fermions
    from QES.Algebra.Operator                   import operators_hardcore

class OperatorModule:
    """
    Lazy loader for operator modules based on LocalSpace type.
    
    Provides attribute access to operator factory functions without requiring
    upfront imports. Modules are imported only when first accessed.
    """
    
    def __init__(self, local_space_type: Optional[LocalSpaceTypes] = None):
        """
        Initialize the operator module loader.
        
        Parameters
        ----------
        local_space_type : LocalSpaceTypes, optional
            The type of local space to determine which operators to load.
            If None, defaults to SPIN_HALF.
        """
        self._local_space_type  = local_space_type or LocalSpaceTypes.SPIN_1_2
        self._spin_module       = None
        self._spin1_module      = None
        self._fermion_module    = None
        self._hardcore_module   = None
        self._anyon_module      = None

    def _load_spin_operators(self):
        """Lazy load spin-1/2 operator module."""
        if self._spin_module is None:
            from QES.Algebra.Operator import operators_spin
            self._spin_module = operators_spin
        return self._spin_module
    
    def _load_spin1_operators(self):
        """Lazy load spin-1 operator module."""
        if self._spin1_module is None:
            from QES.Algebra.Operator import operators_spin_1
            self._spin1_module = operators_spin_1
        return self._spin1_module
    
    def _load_fermion_operators(self):
        """Lazy load spinless fermion operator module."""
        if self._fermion_module is None:
            from QES.Algebra.Operator import operators_spinless_fermions
            self._fermion_module = operators_spinless_fermions
        return self._fermion_module
    
    def _load_hardcore_operators(self):
        """Lazy load hardcore boson operator module."""
        if self._hardcore_module is None:
            from QES.Algebra.Operator import operators_hardcore
            self._hardcore_module = operators_hardcore
        return self._hardcore_module
    
    def _load_anyon_operators(self):
        """Lazy load anyon operator module."""
        if self._anyon_module is None:
            from QES.Algebra.Operator import operators_anyon
            self._anyon_module = operators_anyon
        return self._anyon_module
    
    def __getattr__(self, name: str) -> Callable[..., 'Operator']:
        """
        Dynamically load and return operator factory functions.
        
        Parameters
        ----------
        name : str
            Name of the operator factory function
            
        Returns
        -------
        Callable
            The operator factory function
            
        Raises
        ------
        AttributeError
            If the operator is not found in any loaded module
        """
        # Determine which module to load based on local space type and operator name
        if self._local_space_type == LocalSpaceTypes.SPIN_1_2:
            # Spin-1/2 operators: sig_x, sig_y, sig_z, sig_plus, sig_minus, etc.
            if name.startswith('sig_') or name.startswith('sigma_'):
                module = self._load_spin_operators()
                if hasattr(module, name):
                    return getattr(module, name)
        
        elif self._local_space_type == LocalSpaceTypes.SPIN_1:
            # Spin-1 operators: s1_x, s1_y, s1_z, s1_plus, s1_minus, etc.
            if name.startswith('s1_') or name.startswith('spin1_'):
                module = self._load_spin1_operators()
                if hasattr(module, name):
                    return getattr(module, name)
            # Also allow sig_ aliases for spin-1 (maps to s1_)
            if name.startswith('sig_'):
                s1_name = name.replace('sig_', 's1_')
                module = self._load_spin1_operators()
                if hasattr(module, s1_name):
                    return getattr(module, s1_name)
        
        elif self._local_space_type == LocalSpaceTypes.SPINLESS_FERMIONS:
            # Fermion operators: c_dag, c_ann, n_op, etc.
            if name in ('c_dag', 'c', 'c_ann', 'n', 'n_op', 'fermion_number'):
                module = self._load_fermion_operators()
                if hasattr(module, name):
                    return getattr(module, name)
        
        elif self._local_space_type == LocalSpaceTypes.BOSONS:
            # Boson operators (includes hardcore bosons)
            if name in ('b_dag', 'b', 'b_ann', 'n', 'n_op'):
                module = self._load_hardcore_operators()
                if hasattr(module, name):
                    return getattr(module, name)
        
        # Fallback: try all modules in order
        for loader in [self._load_spin_operators, 
                       self._load_spin1_operators,
                       self._load_fermion_operators,
                       self._load_hardcore_operators]:
            try:
                module = loader()
                if hasattr(module, name):
                    return getattr(module, name)
            except ImportError:
                continue
        
        # If not found, raise AttributeError
        raise AttributeError(
            f"Operator '{name}' not found for LocalSpace type {self._local_space_type}. "
            f"Available operators depend on the local space type:\n"
            f"  - SPIN_1_2/SPIN_1   : sig_x, sig_y, sig_z, sig_plus, sig_minus, sig_z_total\n"
            f"  - SPINLESS_FERMIONS : c_dag, c (or c_ann), n (or n_op)\n"
            f"  - BOSONS            : b_dag, b (or b_ann), n (or n_op)"
        )
    
    def __dir__(self):
        """
        Return list of available operator names.
        
        Returns
        -------
        list
            List of available operator factory function names
        """
        operators = []
        
        if self._local_space_type == LocalSpaceTypes.SPIN_1_2:
            operators.extend(['sig_x', 'sig_y', 'sig_z', 'sig_plus', 'sig_minus', 
                'sig_z_total', 'sigma_x', 'sigma_y', 'sigma_z'])
        
        elif self._local_space_type == LocalSpaceTypes.SPIN_1:
            operators.extend(['s1_x', 's1_y', 's1_z', 's1_z2', 's1_plus', 's1_minus',
                's1_p', 's1_m', 'spin1_x_int', 'spin1_y_int', 'spin1_z_int'])
        
        elif self._local_space_type == LocalSpaceTypes.SPINLESS_FERMIONS:
            operators.extend(['c_dag', 'c', 'c_ann', 'n', 'n_op', 'fermion_number'])
        
        elif self._local_space_type == LocalSpaceTypes.BOSONS:
            operators.extend(['b_dag', 'b', 'b_ann', 'n', 'n_op'])
        
        return operators
    
    # --------------------------------------------------------------
    #! Instruction codes and function
    # --------------------------------------------------------------
    
    def get_codes(self, is_complex: bool = False) -> None:
        '''
        Set up instruction codes and composition function based on physics type.
        
        This method configures the lookup codes, instruction function, and
        maximum output size for efficient JIT-compiled composition.
        '''
        
        if self._local_space_type == LocalSpaceTypes.SPIN_1_2:
            spin_module            = self._load_spin_operators()
            self._lookup_codes     = spin_module.SPIN_LOOKUP_CODES.to_dict()
            self._instr_function   = spin_module.sigma_composition_integer(is_complex=is_complex)
        elif self._local_space_type == LocalSpaceTypes.SPIN_1:
            spin1_module           = self._load_spin1_operators()
            self._lookup_codes     = spin1_module.SPIN1_LOOKUP_CODES.to_dict()
            self._instr_function   = spin1_module.spin1_composition_integer(is_complex=is_complex)
        elif self._local_space_type == LocalSpaceTypes.SPINLESS_FERMIONS:
            fermion_module         = self._load_fermion_operators()
            self._lookup_codes     = fermion_module.FERMION_LOOKUP_CODES.to_dict()
            self._instr_function   = fermion_module.fermion_composition_integer(is_complex=is_complex)
        elif self._local_space_type == LocalSpaceTypes.BOSONS:
            raise NotImplementedError("Hardcore boson instruction codes not yet implemented.")
        else:
            raise ValueError(f"Unknown local space type: {self._local_space_type}")
        return self._lookup_codes, self._instr_function
    
    def get_composition(self, is_cpx: bool = True, *, custom_op_funcs: Optional[List]=None, custom_op_arity: Optional[int]=None) -> Optional[Callable]:
        '''
        Get the JIT-compiled composition function based on physics type.
        
        Parameters
        ----------
        custom : bool, optional
            Whether to use custom operator composition. Default is False.
        
        Returns
        -------
        Callable or None
            The hybrid composition function.
        '''
        custom = custom_op_funcs is not None and custom_op_arity is not None

        if self._local_space_type == LocalSpaceTypes.SPIN_1_2:
            spin_module    = self._load_spin_operators()
            if custom:
                return spin_module.sigma_composition_with_custom(is_complex=is_cpx, custom_op_funcs=custom_op_funcs, custom_op_arity=custom_op_arity)
            else:
                return spin_module.sigma_composition_integer(is_complex=is_cpx)
        
        elif self._local_space_type == LocalSpaceTypes.SPIN_1:
            spin1_module   = self._load_spin1_operators()
            # Spin-1 doesn't have custom composition yet - use standard
            return spin1_module.spin1_composition_integer(is_complex=is_cpx)
            
        elif self._local_space_type == LocalSpaceTypes.SPINLESS_FERMIONS:
            fermion_module = self._load_fermion_operators()
            if custom:
                return fermion_module.fermion_composition_with_custom(is_complex=is_cpx, custom_op_funcs=custom_op_funcs, custom_op_arity=custom_op_arity)
            else:
                return fermion_module.fermion_composition_integer(is_complex=is_cpx)
            
        elif self._local_space_type == LocalSpaceTypes.BOSONS:
            raise NotImplementedError("Hardcore boson composition function not yet implemented.")
        
        elif self._local_space_type == LocalSpaceTypes.ANYONS:
            raise NotImplementedError("Anyon composition function not yet implemented.")
        
        else:
            raise ValueError(f"Unknown local space type: {self._local_space_type}")
        
        return None
    
    # --------------------------------------------------------------
    # Utility method to print help
    # --------------------------------------------------------------
    
    def help(self):
        """
        Print help message showing available operators for current local space type.
        """
        print(f"Operator Module for LocalSpace: {self._local_space_type}")
        print("=" * 70)
        print("\nAvailable operators:")
        
        if self._local_space_type in (LocalSpaceTypes.SPIN_1_2, LocalSpaceTypes.SPIN_1):
            print("\n  Spin Operators:")
            print("    sig_x        - Pauli X (sigma_x) operator")
            print("    sig_y        - Pauli Y (sigma_y) operator") 
            print("    sig_z        - Pauli Z (sigma_z) operator")
            print("    sig_plus     - Raising operator (sigma_+)")
            print("    sig_minus    - Lowering operator (sigma_-)")
            print("    sig_z_total  - Total magnetization (sigma_i sigma_z^i)")
            
        elif self._local_space_type == LocalSpaceTypes.SPINLESS_FERMIONS:
            print("\n  Fermionic Operators:")
            print("    c_dag        - Creation operator (c^+)")
            print("    c, c_ann     - Annihilation operator (c)")
            print("    n, n_op      - Number operator (n = c^+c)")
            
        elif self._local_space_type == LocalSpaceTypes.BOSONS:
            print("\n  Boson Operators:")
            print("    b_dag        - Creation operator (b^+)")
            print("    b, b_ann     - Annihilation operator (b)")
            print("    n, n_op      - Number operator (n = b^+b)")
        
        print("\nUsage:")
        print("  op = operators.sig_x(ns=4, sites=[0, 1])")
        print("  matrix = op.matrix")
        print("=" * 70)    
    
    # --------------------------------------------------------------
    #! Correlation Operators Generation Method
    # --------------------------------------------------------------
    
    def correlators(self, *, indices_pairs: Optional[list] = None, correlators = None, type_acting = 'global', **kwargs):
        """
        Generate correlation operators for specified correlator types and site indices.
        Parameters
        ----------
        indices_pairs : list of tuples or None, optional
            List of (i, j) index tuples specifying the sites for the correlators. If None, defaults to just correlators...
        correlators : list or None, optional
            List of correlator types to generate (e.g., ['zz', 'xx', 'xy', ...]). If None, defaults to ['zz'] for spin systems.
        type_acting : str, optional
            Specifies whether the operators act 'locally' on specified sites or 'globally' on the entire system. Default is 'global'.
            It can be:
            - 'local'       : Operators act only on the specified sites.
            - 'global'      : Operators act on the entire system, with identity on other sites.
            - 'correlation' : Specialized for correlation functions.
            If indices pairs are specified, this parameter is redundant. Otherwise,
            it determines how the operators are constructed.
        **kwargs
            Additional keyword arguments passed to the operator constructors.
        Returns
        -------
        dict
            Dictionary mapping (i, j) tuples or 'i,j' to the corresponding operator objects for each correlator type.
        Raises
        ------
        ValueError
            If an unknown correlator type or module is specified.
        NotImplementedError
            If correlators for fermions or hardcore bosons are requested (not yet implemented).
        Notes
        -----
        - Supported correlator types for spin systems include: 'xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz'.
        - For unsupported local space types, an error is raised.
        """

        # we can return operators with indices applied
        if indices_pairs is None:
            indices_pairs   = [(None, None)]
        
        # Pass type_acting to kwargs
        kwargs['type_act']  = type_acting
        
        if self._local_space_type == LocalSpaceTypes.SPIN_1_2:
            ops_module  = self._load_spin_operators()
            correlators = correlators or ['zz']
            ops         = { x : {} for x in correlators }
            for corr in correlators:
                for (i, j) in indices_pairs:
                    sites               = [i,j] if i is not None and j is not None else None
                    key                 = (i,j) if sites is not None else 'i,j'
                    kwargs_op           = kwargs.copy()
                    kwargs_op['sites']  = sites
                    
                    if 'xx' == corr:
                        op = ops_module.sig_x(**kwargs_op)
                    elif 'xy' == corr:
                        op = ops_module.sig_xy(**kwargs_op)
                    elif 'xz' == corr:
                        op = ops_module.sig_xz(**kwargs_op)
                    elif 'yx' == corr:
                        op = ops_module.sig_yx(**kwargs_op)
                    elif 'yy' == corr:
                        op = ops_module.sig_y(**kwargs_op)
                    elif 'yz' == corr:
                        op = ops_module.sig_yz(**kwargs_op)
                    elif 'zx' == corr:
                        op = ops_module.sig_zx(**kwargs_op)
                    elif 'zy' == corr:
                        op = ops_module.sig_zy(**kwargs_op)
                    elif 'zz' == corr:
                        op = ops_module.sig_z(**kwargs_op)
                    else:
                        raise ValueError(f"Unknown correlator type: {corr}")
                    ops[corr][key] = op
            return ops
    
        elif self._local_space_type == LocalSpaceTypes.SPIN_1:
            ops_module  = self._load_spin1_operators()
            correlators = correlators or ['zz']
            ops         = { x : {} for x in correlators }
            for corr in correlators:
                for (i, j) in indices_pairs:
                    sites               = [i,j] if i is not None and j is not None else None
                    key                 = (i,j) if sites is not None else 'i,j'
                    kwargs_op           = kwargs.copy()
                    kwargs_op['sites']  = sites
                    
                    if 'xx' == corr:
                        op = ops_module.s1_x(**kwargs_op)
                    elif 'xy' == corr:
                        op = ops_module.s1_xy(**kwargs_op)
                    elif 'xz' == corr:
                        op = ops_module.s1_xz(**kwargs_op)
                    elif 'yx' == corr:
                        op = ops_module.s1_yx(**kwargs_op)
                    elif 'yy' == corr:
                        op = ops_module.s1_y(**kwargs_op)
                    elif 'yz' == corr:
                        op = ops_module.s1_yz(**kwargs_op)
                    elif 'zx' == corr:
                        op = ops_module.s1_zx(**kwargs_op)
                    elif 'zy' == corr:
                        op = ops_module.s1_zy(**kwargs_op)
                    elif 'zz' == corr:
                        op = ops_module.s1_z(**kwargs_op)
                    else:
                        raise ValueError(f"Unknown correlator type: {corr}")
                    ops[corr][key] = op
            return ops
        
        elif self._local_space_type == LocalSpaceTypes.SPINLESS_FERMIONS:
            ops_module = self._load_fermion_operators()
            raise NotImplementedError("Fermion correlators not yet implemented.")
        
        elif self._local_space_type == LocalSpaceTypes.BOSONS:
            ops_module = self._load_hardcore_operators()
            raise NotImplementedError("Hardcore boson correlators not yet implemented.")
        else:
            raise ValueError(f"Unknown module: {self._local_space_type}")

    def _compute_spin_correlations(self, 
                                eigenvectors            : np.ndarray,
                                eigenvalues             : np.ndarray,
                                hilbert                 : 'HilbertSpace',
                                lattice                 : Optional['Lattice']   = None,
                                *,
                                nstates_to_store        : Optional[int]         = None,
                                correlators             : Optional[list]        = None,
                                logger                  : Optional['Logger']    = None,
                                n_susceptibility_states : int                   = 10,
                                safety_factor           : float                 = 0.6) -> Dict[str, np.ndarray]:
        """
        Computes Spin-Spin correlations, Total Magnetization projections, 
        Fidelity Susceptibility, and Static Susceptibility in a MEMORY-SAFE way.

        It automatically batches the eigenvectors to fit into available RAM.

        Parameters
        ----------
        eigenvectors : np.ndarray
            The eigenvectors (N_hilbert, N_states).
        eigenvalues : np.ndarray
            The eigenvalues (N_states,).
        hilbert : HilbertSpace
            The Hilbert space for operator generation.
        lattice : Optional['Lattice']
            The lattice object containing site information (.ns). If None, must be inferable from hilbert.
        nstates_to_store : int, optional
            Number of eigenstates to consider. If None, uses all available states.
        correlators : list, optional
            List of correlators to compute (e.g., ['xx', 'xy', 'zz']). If None, computes all.
        logger : Logger, optional
            Logger for progress updates.
        n_susceptibility_states : int
            Number of low-lying states to compute fidelity susceptibility for.
        safety_factor : float
            Fraction of available RAM to utilize (0.0 to 1.0). Lower is safer.

        Returns
        -------
        Dict
            Dictionary containing 'xx', 'xy'..., 'susceptibility_tensor', etc.
        """
        import gc
        import numba

        # Setup & Dimensions
        ns                  = hilbert.ns
        nh, n_total         = eigenvectors.shape
        n_states            = nstates_to_store if nstates_to_store is not None else n_total
        n_threads           = numba.config.NUMBA_NUM_THREADS
        numba.set_num_threads(n_threads)
        
        # Convert to complex once if not already
        ev                  = eigenvectors[:, :n_states]    # (Nh, n_states)
        ev_c_t              = ev.conj().T                   # (N_states, Nh) - usually fits in RAM if EV fits
        
        # Dynamic Batch Size Estimation
        # Size of one full Hilbert vector in GB
        vec_gb              = (nh * 16) / (1024**3) 
        
        # Per batch index, we hold:
        # 1. ev_batch
        # 2. vec_xi, vec_yi, vec_zi (Site i ops)
        # 3. vec_xj, vec_yj, vec_zj (Site j ops)
        # 4. 1-2 Temporary arrays created by numpy during (conj * vec) multiplication
        vecs_per_batch_idx  = 10.0
        avail_mem           = get_available_memory_gb()             # GB
        safe_mem            = avail_mem * safety_factor             # GB to use safely
        
        # Calculate how many states we can process at once
        batch_size          = int(safe_mem / (vec_gb * vecs_per_batch_idx))
        batch_size          = max(1, min(batch_size, n_states)) # Clamp between 1 and total states
        
        if logger:
            logger.info(f"Memory Check: Available={avail_mem:.2f}GB. Vector Size={vec_gb:.4f}GB.",  lvl=2, color='cyan')
            logger.info(f"Calculated safe batch size: {batch_size} states (out of {n_states}).",    lvl=3, color='green')

        # Initialize Data Structures
        ops_list            = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz'] if correlators is None else correlators
        ops_set             = set(ops_list)                         # Fast lookup
        values              = {op: np.zeros((ns, ns, n_states), dtype=np.complex128) for op in ops_list}
        
        # Projectors for total magnetization components. This 
        Sx_proj             = np.zeros((n_states, n_states), dtype=np.complex128)
        Sy_proj             = np.zeros((n_states, n_states), dtype=np.complex128)
        Sz_proj             = np.zeros((n_states, n_states), dtype=np.complex128)
        
        # Load Operators (Local)
        # Note: We create them once. JIT compilation happens on first call.
        sig_x               = self.sig_x(lattice=lattice, ns=ns, type_act='local')
        sig_y               = self.sig_y(lattice=lattice, ns=ns, type_act='local')
        sig_z               = self.sig_z(lattice=lattice, ns=ns, type_act='local')
        
        # Site j: ONLY needed if specific pairwise correlations are requested
        # We look strictly at the second character of the operator string (e.g., 'xz' -> z on j)
        need_xj             = not ops_set.isdisjoint({'xx', 'yx', 'zx'}) 
        need_yj             = not ops_set.isdisjoint({'xy', 'yy', 'zy'})
        need_zj             = not ops_set.isdisjoint({'xz', 'yz', 'zz'})
        
        # Site i Buffers (Always allocate)
        buf_xi              = np.zeros((nh, batch_size), dtype=np.complex128)
        buf_yi              = np.zeros((nh, batch_size), dtype=np.complex128)
        buf_zi              = np.zeros((nh, batch_size), dtype=np.complex128)
        
        # Site j Buffers (Conditionally allocate)
        buf_xj              = np.zeros((nh, batch_size), dtype=np.complex128) if need_xj else None
        buf_yj              = np.zeros((nh, batch_size), dtype=np.complex128) if need_yj else None
        buf_zj              = np.zeros((nh, batch_size), dtype=np.complex128) if need_zj else None
        JIT_CHUNK_SIZE      = 8
        thread_buf_cache    = np.zeros((n_threads, nh, JIT_CHUNK_SIZE), dtype=np.complex128)
        
        # Main Computation Loop (Batched)
        # Outer loop: Sites i
        for i in range(ns):
            t_site      = time.time()
            if logger:  logger.info(f"Computing correlations for site {i+1}/{ns}...", lvl=2)
            
            # Batched Loop over Eigenstates
            for b_start in range(0, n_states, batch_size):
                # Slice the batch
                # Shape: (Nh, current_batch_size)
                b_end                       = min(b_start + batch_size, n_states)
                ev_batch                    = ev[:, b_start:b_end] 
                curr_width                  = b_end - b_start
                
                # Compute Site i operators on this batch
                vec_xi, vec_yi, vec_zi      = None, None, None
                
                if buf_xi is not None:
                    vec_xi = buf_xi[:, :curr_width]; vec_xi.fill(0)
                    sig_x.matvec(ev_batch, i, hilbert=hilbert, out=vec_xi, thread_buffer=thread_buf_cache, chunk_size=JIT_CHUNK_SIZE)
                
                if buf_yi is not None:
                    vec_yi = buf_yi[:, :curr_width]; vec_yi.fill(0)
                    sig_y.matvec(ev_batch, i, hilbert=hilbert, out=vec_yi, thread_buffer=thread_buf_cache, chunk_size=JIT_CHUNK_SIZE)
                if buf_zi is not None:
                    vec_zi = buf_zi[:, :curr_width]; vec_zi.fill(0)
                    sig_z.matvec(ev_batch, i, hilbert=hilbert, out=vec_zi, thread_buffer=thread_buf_cache, chunk_size=JIT_CHUNK_SIZE)

                # Update Projections
                if vec_xi is not None       : Sx_proj[:, b_start:b_end] += ev_c_t @ vec_xi
                if vec_yi is not None       : Sy_proj[:, b_start:b_end] += ev_c_t @ vec_yi
                if vec_zi is not None       : Sz_proj[:, b_start:b_end] += ev_c_t @ vec_zi
                
                # Pre-conjugate for inner loop dot products
                vec_xi_c                    = vec_xi.conj() if vec_xi is not None else None
                vec_yi_c                    = vec_yi.conj() if vec_yi is not None else None
                vec_zi_c                    = vec_zi.conj() if vec_zi is not None else None

                for j in range(i, ns):
                    
                    vec_xj, vec_yj, vec_zj  = None, None, None

                    # Optimization: Point to existing 'i' vectors if i==j
                    if i == j:
                        vec_xj, vec_yj, vec_zj  = vec_xi, vec_yi, vec_zi
                    else:
                        # Otherwise compute into 'j' buffers
                        if buf_xj is not None:
                            vec_xj              = buf_xj[:, :curr_width]; vec_xj.fill(0)
                            sig_x.matvec(ev_batch, j, hilbert=hilbert, out=vec_xj, thread_buffer=thread_buf_cache, chunk_size=JIT_CHUNK_SIZE)
                        
                        if buf_yj is not None:
                            vec_yj              = buf_yj[:, :curr_width]; vec_yj.fill(0)
                            sig_y.matvec(ev_batch, j, hilbert=hilbert, out=vec_yj, thread_buffer=thread_buf_cache, chunk_size=JIT_CHUNK_SIZE)
                            
                        if buf_zj is not None:
                            vec_zj              = buf_zj[:, :curr_width]; vec_zj.fill(0)
                            sig_z.matvec(ev_batch, j, hilbert=hilbert, out=vec_zj, thread_buffer=thread_buf_cache, chunk_size=JIT_CHUNK_SIZE)
                    # Compute Expectations
                    # Inner product: sum(v_i.conj * v_j)
                    
                    if 'xx' in ops_set and vec_xi_c is not None and vec_xj is not None:
                        val                                 = np.sum(vec_xi_c * vec_xj, axis=0) * 4
                        values['xx'][i, j, b_start:b_end]   = val
                        values['xx'][j, i, b_start:b_end]   = val.conj()

                    if 'xy' in ops_set and vec_xi_c is not None and vec_yj is not None:
                        val                                 = np.sum(vec_xi_c * vec_yj, axis=0) * 4
                        values['xy'][i, j, b_start:b_end]   = val
                        values['xy'][j, i, b_start:b_end]   = val.conj()

                    if 'xz' in ops_set and vec_xi_c is not None and vec_zj is not None:
                        val                                 = np.sum(vec_xi_c * vec_zj, axis=0) * 4
                        values['xz'][i, j, b_start:b_end]   = val
                        values['xz'][j, i, b_start:b_end]   = val.conj()
                    
                    if 'yx' in ops_set and vec_yi_c is not None and vec_xj is not None:
                        val                                 = np.sum(vec_yi_c * vec_xj, axis=0) * 4
                        values['yx'][i, j, b_start:b_end]   = val
                        values['yx'][j, i, b_start:b_end]   = val.conj()
                    if 'yy' in ops_set and vec_yi_c is not None and vec_yj is not None:
                        val                                 = np.sum(vec_yi_c * vec_yj, axis=0) * 4
                        values['yy'][i, j, b_start:b_end]   = val
                        values['yy'][j, i, b_start:b_end]   = val.conj()
                    if 'yz' in ops_set and vec_yi_c is not None and vec_zj is not None:
                        val                                 = np.sum(vec_yi_c * vec_zj, axis=0) * 4
                        values['yz'][i, j, b_start:b_end]   = val
                        values['yz'][j, i, b_start:b_end]   = val.conj()
                        
                    if 'zx' in ops_set and vec_zi_c is not None and vec_xj is not None:
                        val                                 = np.sum(vec_zi_c * vec_xj, axis=0) * 4
                        values['zx'][i, j, b_start:b_end]   = val
                        values['zx'][j, i, b_start:b_end]   = val.conj()
                    if 'zy' in ops_set and vec_zi_c is not None and vec_yj is not None:
                        val                                 = np.sum(vec_zi_c * vec_yj, axis=0) * 4
                        values['zy'][i, j, b_start:b_end]   = val
                        values['zy'][j, i, b_start:b_end]   = val.conj()
                    if 'zz' in ops_set and vec_zi_c is not None and vec_zj is not None:
                        val                                 = np.sum(vec_zi_c * vec_zj, axis=0) * 4
                        values['zz'][i, j, b_start:b_end]   = val
                        values['zz'][j, i, b_start:b_end]   = val.conj()
            
            # End of batch loop
            gc.collect()
            
            if logger: 
                logger.info(f"Site {i+1} done in {time.time()-t_site:.2f}s", lvl=3)
                log_memory_status(f"Post-Site {i+1}", logger, lvl=4)
        
        # Post-Processing: Susceptibilities
        if logger: logger.info("Computing susceptibilities...", lvl=2)
        
        # 5a. Fidelity Susceptibility
        mag_x                   = np.diag(Sx_proj) / ns
        mag_y                   = np.diag(Sy_proj) / ns
        mag_z                   = np.diag(Sz_proj) / ns
        values['magnetization'] = {'x': mag_x, 'y': mag_y, 'z': mag_z}
        
        n_susceptibility_states = min(n_susceptibility_states, n_states)
        if n_susceptibility_states > 0:
            logger.info(f"Computing fidelity susceptibility for {n_susceptibility_states} states...", lvl=2)
            
            try:
                from QES.Algebra.Properties.statistical import fidelity_susceptibility_low_rank
            except ImportError:
                pass

            mu_val      = ns / np.sqrt(nh) # Scaling factor for fidelity susceptibility
            fids        = {'x': [], 'y': [], 'z': [], 'tot': []}

            for k in range(min(n_states, n_susceptibility_states)):
                fs_x        = fidelity_susceptibility_low_rank(ev, Sx_proj,                     mu=mu_val, idx=k)
                fs_y        = fidelity_susceptibility_low_rank(ev, Sy_proj,                     mu=mu_val, idx=k)
                fs_z        = fidelity_susceptibility_low_rank(ev, Sz_proj,                     mu=mu_val, idx=k)
                fs_total    = fidelity_susceptibility_low_rank(ev, Sx_proj + Sy_proj + Sz_proj, mu=mu_val, idx=k)
                fids['x'].append(fs_x)
                fids['y'].append(fs_y)
                fids['z'].append(fs_z)
                fids['tot'].append(fs_total)
                
            if logger:
                logger.info("Fidelity susceptibility computation completed.", lvl=3)
            
            values['fidelity_susceptibility'] = {
                'susceptibility/fid/x'      : fids['x'],
                'susceptibility/fid/y'      : fids['y'],
                'susceptibility/fid/z'      : fids['z'],
                'susceptibility/fid/tot'    : fids['tot'],
            }

        # Static Susceptibility Tensor
        chi         = np.zeros((3,3), dtype=np.complex128)
        ops_proj    = [Sx_proj, Sy_proj, Sz_proj]
        E0          = eigenvalues[0]
        
        # Sum over excited states (n > 0)
        # chi_ab = sum_{n!=0} <0|A|n><n|B|0> / (En - E0)
        # ops_proj[a][0, n] is <0|S_a|n>
        for a in range(3):
            for b in range(3):
                num = 0.0j
                for n in range(1, n_states):
                    denom = eigenvalues[n] - E0
                    if np.abs(denom) > 1e-12:
                        term = ops_proj[a][0, n] * ops_proj[b][n, 0] / denom
                        num += term
                chi[a, b] = num
        
        values['susceptibility_tensor'] = chi
        
        if logger: logger.info("Correlation analysis completed.", lvl=2, color='green')
        return values
    
    def compute_correlations(self, 
                            eigenvectors                : np.ndarray,
                            eigenvalues                 : np.ndarray,
                            hilbert                     : 'HilbertSpace',
                            lattice                     : Optional[object]      = None,
                            *,
                            nstates_to_store            : Optional[int]         = None,
                            correlators                 : Optional[list]        = None,
                            logger                      : Optional['Logger']    = None,
                            n_susceptibility_states     : int                   = 10,
                            safety_factor               : float                 = 0.6) -> Dict[str, np.ndarray]:
        """
        Compute correlation functions using the operator module.
        
        Parameters
        ----------
        eigenvectors : np.ndarray
            The eigenvectors (N_hilbert, N_states).
        eigenvalues : np.ndarray
            The eigenvalues (N_states,).
        lattice : Lattice
            The lattice object containing site information (.ns).
        hilbert : HilbertSpace
            The Hilbert space for operator generation.
        logger : Logger, optional
            Logger for progress updates.
        n_susceptibility_states : int
            Number of low-lying states to compute fidelity susceptibility for.
        safety_factor : float
            Fraction of available RAM to utilize (0.0 to 1.0). Lower is safer.
            
        Returns
        -------
        Dict
            Dictionary containing correlation results.
            1. Spin systems: 'xx', 'xy'..., 'susceptibility_tensor', etc.
                {'xx': np.ndarray, 'xy': np.ndarray, ..., 'susceptibility_tensor': np.ndarray, ...}
            
        Raises
        ------
        NotImplementedError
            If correlation computations for the local space type are not implemented.
        """
        if self._local_space_type in (LocalSpaceTypes.SPIN_1_2, LocalSpaceTypes.SPIN_1):
            return self._compute_spin_correlations(
                eigenvectors                = eigenvectors,
                eigenvalues                 = eigenvalues,
                lattice                     = lattice,
                hilbert                     = hilbert,
                nstates_to_store            = nstates_to_store,
                correlators                 = correlators,
                logger                      = logger,
                n_susceptibility_states     = n_susceptibility_states,
                safety_factor               = safety_factor
            )
        else:
            raise NotImplementedError("Correlation computations for this local space type are not yet implemented.")
    
    # --------------------------------------------------------------
    # Static utility methods for common operations
    # --------------------------------------------------------------

    @staticmethod
    def overlap(a, O, b, backend: str = 'default'):
        """
        Compute the overlap <a|O|b> using the specified backend.
        """
        from QES.Algebra.backends import overlap
        return overlap(a, O, b, backend=backend)
    
    @staticmethod
    def kron(A, B, backend: str = 'default'):
        """
        Compute the Kronecker product A ⊗ B using the specified backend.
        """
        from QES.Algebra.backends import kron
        return kron(A, B, backend=backend)
    
    @staticmethod
    def outer(a, b, backend: str = 'default'):
        """
        Compute the outer product |a><b| using the specified backend.
        """
        from QES.Algebra.backends import outer
        return outer(a, b, backend=backend)

    # --------------------------------------------------------------

def get_operator_module(local_space_type: Optional[LocalSpaceTypes] = None) -> OperatorModule:
    """
    Get an OperatorModule for the specified local space type.
    
    Parameters
    ----------
    local_space_type : LocalSpaceTypes, optional
        The type of local space. If None, defaults to SPIN_1_2.
        
    Returns
    -------
    OperatorModule
        Lazy loader for operator factory functions
        
    Examples
    --------
    >>> ops     = get_operator_module(LocalSpaceTypes.SPIN_1_2)
    >>> sig_x   = ops.sig_x(ns=4, sites=[0])
    """
    return OperatorModule(local_space_type)

# ============================================================================
#! END OF FILE
# ============================================================================
