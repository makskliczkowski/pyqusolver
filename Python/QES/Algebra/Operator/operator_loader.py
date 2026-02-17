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

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import numpy as np

try:
    from QES.Algebra.Hilbert.hilbert_local import LocalSpaceTypes, choose_chunk_size
    from QES.general_python.common.memory import get_available_memory_gb, log_memory_status
except ImportError as e:
    raise ImportError(
        "Could not import LocalSpaceTypes. Ensure that QES package is properly installed."
    ) from e

if TYPE_CHECKING:
    from QES.Algebra.hilbert import HilbertSpace
    from QES.Algebra.Operator import (
        Operator,
    )
    from QES.general_python.common.flog import Logger
    from QES.general_python.lattices.lattice import Lattice

# --------------------------------------------------------------


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
        self._local_space_type = local_space_type or LocalSpaceTypes.SPIN_1_2
        self._spin_module = None
        self._spin1_module = None
        self._fermion_module = None
        self._hardcore_module = None
        self._anyon_module = None

        # Cache for correlation operators to avoid JIT recompilation
        self._correlation_ops_cache = {}

    def _load_spin_operators(self):
        """Lazy load spin-1/2 operator module."""
        if self._spin_module is None:
            from QES.Algebra.Operator.impl import operators_spin

            self._spin_module = operators_spin
        return self._spin_module

    def _load_spin1_operators(self):
        """Lazy load spin-1 operator module."""
        if self._spin1_module is None:
            from QES.Algebra.Operator.impl import operators_spin_1

            self._spin1_module = operators_spin_1
        return self._spin1_module

    def _load_fermion_operators(self):
        """Lazy load spinless fermion operator module."""
        if self._fermion_module is None:
            from QES.Algebra.Operator.impl import operators_spinless_fermions

            self._fermion_module = operators_spinless_fermions
        return self._fermion_module

    def _load_hardcore_operators(self):
        """Lazy load hardcore boson operator module."""
        if self._hardcore_module is None:
            from QES.Algebra.Operator.impl import operators_hardcore

            self._hardcore_module = operators_hardcore
        return self._hardcore_module

    def _load_anyon_operators(self):
        """Lazy load anyon operator module."""
        if self._anyon_module is None:
            from QES.Algebra.Operator.impl import operators_anyon

            self._anyon_module = operators_anyon
        return self._anyon_module

    def __getattr__(self, name: str) -> Callable[..., "Operator"]:
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
            if name.startswith("sig_") or name.startswith("sigma_"):
                module = self._load_spin_operators()
                if hasattr(module, name):
                    return getattr(module, name)

        elif self._local_space_type == LocalSpaceTypes.SPIN_1:
            # Spin-1 operators: s1_x, s1_y, s1_z, s1_plus, s1_minus, etc.
            if name.startswith("s1_") or name.startswith("spin1_"):
                module = self._load_spin1_operators()
                if hasattr(module, name):
                    return getattr(module, name)
            # Also allow sig_ aliases for spin-1 (maps to s1_)
            if name.startswith("sig_"):
                s1_name = name.replace("sig_", "s1_")
                module = self._load_spin1_operators()
                if hasattr(module, s1_name):
                    return getattr(module, s1_name)

        elif self._local_space_type == LocalSpaceTypes.SPINLESS_FERMIONS:
            # Fermion operators: cdag, c, n_op, etc.
            if name == "c_dag":
                name = "cdag"
            if name in ("cdag", "c", "c_ann", "n", "n_op", "fermion_number"):
                module = self._load_fermion_operators()
                if hasattr(module, name):
                    return getattr(module, name)

        elif self._local_space_type == LocalSpaceTypes.BOSONS:
            # Boson operators (includes hardcore bosons)
            if name in ("b_dag", "b", "b_ann", "n", "n_op"):
                module = self._load_hardcore_operators()
                if hasattr(module, name):
                    return getattr(module, name)

        # Fallback: try all modules in order
        for loader in [
            self._load_spin_operators,
            self._load_spin1_operators,
            self._load_fermion_operators,
            self._load_hardcore_operators,
        ]:
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
            operators.extend(
                [
                    "sig_x",
                    "sig_y",
                    "sig_z",
                    "sig_plus",
                    "sig_minus",
                    "sig_z_total",
                    "sigma_x",
                    "sigma_y",
                    "sigma_z",
                ]
            )

        elif self._local_space_type == LocalSpaceTypes.SPIN_1:
            operators.extend(
                [
                    "s1_x",
                    "s1_y",
                    "s1_z",
                    "s1_z2",
                    "s1_plus",
                    "s1_minus",
                    "s1_p",
                    "s1_m",
                    "spin1_x_int",
                    "spin1_y_int",
                    "spin1_z_int",
                ]
            )

        elif self._local_space_type == LocalSpaceTypes.SPINLESS_FERMIONS:
            operators.extend(["c_dag", "c", "c_ann", "n", "n_op", "fermion_number"])

        elif self._local_space_type == LocalSpaceTypes.BOSONS:
            operators.extend(["b_dag", "b", "b_ann", "n", "n_op"])

        return operators

    # --------------------------------------------------------------
    #! Instruction codes and function
    # --------------------------------------------------------------

    def get_codes(self, is_complex: bool = False) -> None:
        """
        Set up instruction codes and composition function based on physics type.

        This method configures the lookup codes, instruction function, and
        maximum output size for efficient JIT-compiled composition.
        """

        if self._local_space_type == LocalSpaceTypes.SPIN_1_2:
            spin_module = self._load_spin_operators()
            self._lookup_codes = spin_module.SPIN_LOOKUP_CODES.to_dict()
            self._instr_function = spin_module.sigma_composition_integer(is_complex=is_complex)
        elif self._local_space_type == LocalSpaceTypes.SPIN_1:
            spin1_module = self._load_spin1_operators()
            self._lookup_codes = spin1_module.SPIN1_LOOKUP_CODES.to_dict()
            self._instr_function = spin1_module.spin1_composition_integer(is_complex=is_complex)
        elif self._local_space_type == LocalSpaceTypes.SPINLESS_FERMIONS:
            fermion_module = self._load_fermion_operators()
            self._lookup_codes = fermion_module.FERMION_LOOKUP_CODES.to_dict()
            self._instr_function = fermion_module.fermion_composition_integer(is_complex=is_complex)
        elif self._local_space_type == LocalSpaceTypes.BOSONS:
            raise NotImplementedError("Hardcore boson instruction codes not yet implemented.")
        else:
            raise ValueError(f"Unknown local space type: {self._local_space_type}")
        return self._lookup_codes, self._instr_function

    def get_composition(
        self,
        is_cpx: bool = True,
        *,
        custom_op_funcs: Optional[List] = None,
        custom_op_arity: Optional[int] = None,
    ) -> Optional[Callable]:
        """
        Get the JIT-compiled composition function based on physics type.

        Parameters
        ----------
        custom : bool, optional
            Whether to use custom operator composition. Default is False.

        Returns
        -------
        Callable or None
            The hybrid composition function.
        """
        custom = custom_op_funcs is not None and custom_op_arity is not None

        if self._local_space_type == LocalSpaceTypes.SPIN_1_2:
            spin_module = self._load_spin_operators()
            if custom:
                return spin_module.sigma_composition_with_custom(
                    is_complex=is_cpx,
                    custom_op_funcs=custom_op_funcs,
                    custom_op_arity=custom_op_arity,
                )
            else:
                return spin_module.sigma_composition_integer(is_complex=is_cpx)

        elif self._local_space_type == LocalSpaceTypes.SPIN_1:
            spin1_module = self._load_spin1_operators()
            # Spin-1 doesn't have custom composition yet - use standard
            return spin1_module.spin1_composition_integer(is_complex=is_cpx)

        elif self._local_space_type == LocalSpaceTypes.SPINLESS_FERMIONS:
            fermion_module = self._load_fermion_operators()
            if custom:
                return fermion_module.fermion_composition_with_custom(
                    is_complex=is_cpx,
                    custom_op_funcs=custom_op_funcs,
                    custom_op_arity=custom_op_arity,
                )
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

    def correlators(
        self,
        *,
        indices_pairs: Optional[list] = None,
        correlators=None,
        type_acting="global",
        **kwargs,
    ):
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
            indices_pairs = [(None, None)]

        # Pass type_acting to kwargs
        kwargs["type_act"] = type_acting

        if self._local_space_type == LocalSpaceTypes.SPIN_1_2:
            ops_module = self._load_spin_operators()
            correlators = correlators or ["zz"]
            ops = {x: {} for x in correlators}
            for corr in correlators:
                for i, j in indices_pairs:
                    sites = [i, j] if i is not None and j is not None else None
                    key = (i, j) if sites is not None else "i,j"
                    kwargs_op = kwargs.copy()
                    kwargs_op["sites"] = sites

                    if "xx" == corr:
                        op = ops_module.sig_x(**kwargs_op)
                    elif "xy" == corr:
                        op = ops_module.sig_xy(**kwargs_op)
                    elif "xz" == corr:
                        op = ops_module.sig_xz(**kwargs_op)
                    elif "yx" == corr:
                        op = ops_module.sig_yx(**kwargs_op)
                    elif "yy" == corr:
                        op = ops_module.sig_y(**kwargs_op)
                    elif "yz" == corr:
                        op = ops_module.sig_yz(**kwargs_op)
                    elif "zx" == corr:
                        op = ops_module.sig_zx(**kwargs_op)
                    elif "zy" == corr:
                        op = ops_module.sig_zy(**kwargs_op)
                    elif "zz" == corr:
                        op = ops_module.sig_z(**kwargs_op)
                    else:
                        raise ValueError(f"Unknown correlator type: {corr}")
                    ops[corr][key] = op
            return ops

        elif self._local_space_type == LocalSpaceTypes.SPIN_1:
            ops_module = self._load_spin1_operators()
            correlators = correlators or ["zz"]
            ops = {x: {} for x in correlators}
            for corr in correlators:
                for i, j in indices_pairs:
                    sites = [i, j] if i is not None and j is not None else None
                    key = (i, j) if sites is not None else "i,j"
                    kwargs_op = kwargs.copy()
                    kwargs_op["sites"] = sites

                    if "xx" == corr:
                        op = ops_module.s1_x(**kwargs_op)
                    elif "yy" == corr:
                        op = ops_module.s1_y(**kwargs_op)
                    elif "zz" == corr:
                        op = ops_module.s1_z(**kwargs_op)
                    else:
                        raise ValueError(f"Unsupported spin-1 correlator type '{corr}'. Supported: 'xx', 'yy', 'zz'.")
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

    # ? Computation

    def _compute_spin_correlations(
        self,
        eigenvectors: np.ndarray,
        eigenvalues: np.ndarray,
        hilbert: "HilbertSpace",
        lattice: Optional["Lattice"] = None,
        *,
        nstates_to_store: Optional[int] = None,
        correlators: Optional[list] = None,
        logger: Optional["Logger"] = None,
        # ---
        susc_nstates: int = 10,
        susc_mus: list = [0.01, 0.05, 0.1, 0.5, 1.0],
        safety_factor: float = 0.6,
    ) -> Dict[str, np.ndarray]:
        """
        Computes Spin-Spin correlations, Total Magnetization projections,
        Fidelity Susceptibility, and Static Susceptibility.

        This method uses a unified approach safe for both symmetric and non-symmetric Hilbert spaces.
        Correlations <S_a^i S_b^j> are computed by applying the composite operator (S_a^i S_b^j)
        to the state, then taking the overlap. This ensures correct handling of intermediate states
        in symmetric subspaces.
        """
        import gc
        import time

        import numba

        # Setup & Dimensions
        ns = hilbert.ns
        nh, n_total = eigenvectors.shape
        n_states = nstates_to_store if nstates_to_store is not None else n_total
        n_threads = numba.config.NUMBA_NUM_THREADS
        has_sym = hilbert.has_sym
        numba.set_num_threads(n_threads)

        ev = eigenvectors[:, :n_states]
        ev_c_t = ev.conj().T

        # Determine needed operators
        ops_list = (
            ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]
            if correlators is None
            else correlators
        )
        values = {op: np.zeros((ns, ns, n_states), dtype=np.complex128) for op in ops_list}

        # Components needed for susceptibility/magnetization (single site)
        needed_components = sorted(list(set([c for op in ops_list for c in op if c in "xyz"])))
        ops_module = self._load_spin_operators()

        # Initialize projectors for total magnetization (for susceptibility), correlation operators, etc.
        mag_ops_matrix = {
            comp: np.zeros((n_states, n_states), dtype=np.complex128) for comp in needed_components
        }
        corr_ops = {}

        for op_name in ops_list:
            cache_key = (op_name, ns, "corr")
            if cache_key in self._correlation_ops_cache:
                corr_ops[op_name] = self._correlation_ops_cache[cache_key]
                continue

            if hasattr(ops_module, f"sig_{op_name}"):
                # Use standard factory (e.g. sig_xy)
                factory = getattr(ops_module, f"sig_{op_name}")
                op = factory(ns=ns, sites=None, type_act="corr")
            elif op_name in ("xx", "yy", "zz"):
                # Use sig_x, sig_y, sig_z respectively, but with type_act='corr'
                c1 = op_name[0]
                factory = getattr(ops_module, f"sig_{c1}")
                op = factory(ns=ns, sites=None, type_act="corr")
            else:
                # Fallback for others not in mixed list
                # Use make_nsite_correlator
                c1, c2 = op_name[0], op_name[1]
                op = ops_module.make_nsite_correlator([c1, c2], ns=ns, sites=None)

            self._correlation_ops_cache[cache_key] = op
            corr_ops[op_name] = op

        # Prepare single-site operators for magnetization
        # Also cached for performance
        single_ops = {}
        for comp in ["x", "y", "z"]:
            cache_key = (comp, ns, "local")
            if cache_key in self._correlation_ops_cache:
                single_ops[comp] = self._correlation_ops_cache[cache_key]
            else:
                if comp == "x":
                    op = self.sig_x(ns=ns, type_act="local")
                elif comp == "y":
                    op = self.sig_y(ns=ns, type_act="local")
                elif comp == "z":
                    op = self.sig_z(ns=ns, type_act="local")
                self._correlation_ops_cache[cache_key] = op
                single_ops[comp] = op

        # Memory Management
        vec_gb = (nh * 16) / (1024**3)
        # We need buffers for:
        # 1. Output of correlation op (1 vector per batch)
        # 2. Output of single ops (3 vectors per batch)
        vecs_per_batch = 1.0 + 4.0  # 1 for corr, 3 for single ops
        avail_mem = get_available_memory_gb()
        safe_mem = avail_mem * safety_factor
        batch_size = int(safe_mem / (vec_gb * vecs_per_batch))
        batch_size = max(1, min(batch_size, n_states))

        if logger:
            logger.info(
                f"Correlations: Batch size {batch_size} states (avail: {avail_mem:.1f}GB)",
                lvl=2,
                color="cyan",
            )

        JIT_CHUNK_SIZE = choose_chunk_size(hilbert.nh, n_threads)
        thread_buf_cache = np.zeros((n_threads, nh, JIT_CHUNK_SIZE), dtype=np.complex128)
        buf_corr = np.zeros((nh, batch_size), dtype=np.complex128)

        # Loop over sites
        for i in range(ns):
            t_site = time.time()

            # Loop over batches
            for b_start in range(0, n_states, batch_size):
                b_end = min(b_start + batch_size, n_states)
                curr_width = b_end - b_start
                ev_batch = ev[:, b_start:b_end]

                # Accumulate Magnetization / Susceptibility Terms (Single Site)
                for comp in needed_components:
                    buf = buf_corr[:, :curr_width]
                    buf.fill(0)
                    single_ops[comp].matvec(
                        ev_batch,
                        i,
                        hilbert_in=hilbert,
                        out=buf,
                        thread_buffer=thread_buf_cache,
                        chunk_size=JIT_CHUNK_SIZE,
                    )

                    # Accumulate <n | S_comp^i | m> into total matrix
                    block = ev_c_t @ buf
                    mag_ops_matrix[comp][:, b_start:b_end] += block

                # Loop j
                for j in range(i, ns):
                    # For each requested correlator

                    for op_name in ops_list:
                        buf = buf_corr[:, :curr_width]
                        buf.fill(0)

                        # Diagonal operators (e.g. zz) don't change the state -> use 'fast' kernel O(1)
                        # Off-diagonal operators (xx, xy, ...) may leave symmetry sector → need 'project' O(|G|²)
                        op_modifies = corr_ops[op_name].modifies
                        sym_mode    = 'project' if (has_sym and op_modifies) else ('fast' if has_sym else 'auto')

                        # Apply composite operator S_a^i S_b^j
                        corr_ops[op_name].matvec(
                            ev_batch,
                            i,
                            j,
                            hilbert_in=hilbert,
                            out=buf,
                            thread_buffer=thread_buf_cache,
                            chunk_size=JIT_CHUNK_SIZE,
                            symmetry_mode=sym_mode,
                        )
                        val = np.einsum("bi,ib->b", ev_c_t[b_start:b_end, :], buf)
                        values[op_name][i, j, b_start:b_end] = val

            if logger and (i + 1) % max(1, ns // 5) == 0:
                logger.info(f"Site {i+1}/{ns} done ({time.time()-t_site:.2f}s)", lvl=3)
            gc.collect()

        # Post-Processing: Magnetization
        values["magnetization"] = {}
        for comp in needed_components:
            # Diagonal of total spin matrix / ns
            values["magnetization"][comp] = np.diag(mag_ops_matrix[comp]) / ns

        # Post-Processing: Susceptibilities
        # Fidelity Susceptibility
        susc_nstates = min(susc_nstates, n_states)
        susc_out = {
            c: np.zeros((susc_nstates, len(susc_mus)), dtype=np.float64) for c in needed_components
        }
        susc_out["tot"] = np.zeros((susc_nstates, len(susc_mus)), dtype=np.float64)

        if susc_nstates >= 1:
            try:
                from QES.Algebra.Properties.statistical import fidelity_susceptibility_low_rank
            except ImportError:
                raise ImportError(
                    "fidelity_susceptibility_low_rank not found. Ensure QES package is properly installed."
                )

            proj_tot = sum(mag_ops_matrix.values())
            gaps = np.diff(eigenvalues[: susc_nstates + 1])
            scale = np.median(gaps[gaps > 1e-12]) if np.any(gaps > 1e-12) else np.mean(np.abs(gaps))
            energies = eigenvalues[
                :susc_nstates
            ]  # keep consistent with your "low-rank/subspace" idea

            for imu, mu in enumerate(susc_mus):
                mu_val = mu * scale

                for k in range(susc_nstates):
                    susc_out["tot"][k, imu] = fidelity_susceptibility_low_rank(
                        energies, proj_tot[:susc_nstates, :susc_nstates], mu=mu_val, idx=k
                    )
                    for comp in needed_components:
                        susc_out[comp][k, imu] = fidelity_susceptibility_low_rank(
                            energies,
                            mag_ops_matrix[comp][:susc_nstates, :susc_nstates],
                            mu=mu_val,
                            idx=k,
                        )

            values["fidelity_susceptibility"] = {
                f"susceptibility/fid/{c}": susc_out[c] for c in needed_components
            }
            values["fidelity_susceptibility"]["susceptibility/fid/tot"] = susc_out["tot"]
            values["fidelity_susceptibility"]["susceptibility/fid/mus"] = np.array(susc_mus)
            values["fidelity_susceptibility"]["susceptibility/fid/mu_scale"] = scale

        # Static Susceptibility Tensor
        if len(eigenvalues) > 1:
            chi = np.zeros((3, 3), dtype=np.complex128)
            comp_idx = {"x": 0, "y": 1, "z": 2}
            E0 = eigenvalues[0]

            for a in needed_components:
                for b in needed_components:
                    num = 0.0j
                    Pa = mag_ops_matrix[a]  # <n|Sa|m>
                    Pb = mag_ops_matrix[b]

                    # Sum over n != 0
                    for n in range(1, n_states):
                        denom = eigenvalues[n] - E0
                        if np.abs(denom) > 1e-12:
                            # <0|Sa|n><n|Sb|0>
                            term = Pa[0, n] * Pb[n, 0] / denom
                            num += term

                    chi[comp_idx[a], comp_idx[b]] = num

            values["susceptibility_tensor"] = chi

        if logger:
            logger.info("Correlation analysis completed.", lvl=2, color="green")
        return values

    def compute_correlations(
        self,
        eigenvectors: np.ndarray,
        eigenvalues: np.ndarray,
        hilbert: "HilbertSpace",
        lattice: Optional[object] = None,
        *,
        nstates_to_store: Optional[int] = None,
        correlators: Optional[list] = None,
        logger: Optional["Logger"] = None,
        n_susceptibility_states: int = 10,
        safety_factor: float = 0.6,
    ) -> Dict[str, np.ndarray]:
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
                eigenvectors=eigenvectors,
                eigenvalues=eigenvalues,
                hilbert=hilbert,
                nstates_to_store=nstates_to_store,
                correlators=correlators,
                logger=logger,
                susc_nstates=n_susceptibility_states,
                safety_factor=safety_factor,
            )
        else:
            raise NotImplementedError(
                "Correlation computations for this local space type are not yet implemented."
            )

    # --------------------------------------------------------------
    # Static utility methods for common operations
    # --------------------------------------------------------------

    @staticmethod
    def overlap(a, O, b, backend: str = "default"):
        """
        Compute the overlap <a|O|b> using the specified backend.
        """
        from QES.Algebra.backends import overlap

        return overlap(a, O, b, backend=backend)

    @staticmethod
    def kron(A, B, backend: str = "default"):
        """
        Compute the Kronecker product A ⊗ B using the specified backend.
        """
        from QES.Algebra.backends import kron

        return kron(A, B, backend=backend)

    @staticmethod
    def outer(a, b, backend: str = "default"):
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
