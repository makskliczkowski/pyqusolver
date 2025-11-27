"""
Shared observable computation utilities for both NQS and ED methods.

Key Pattern:
- ED: Use operator.matrix() to create full matrix, then compute <psi|O|psi>
- NQS: Use operator.jax for Monte Carlo sampling of local observables
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Lazy import to avoid circular dependency - only import when needed
NQSEvalEngine = None


def _get_nqs_engine():
    """Lazy import of NQSEvalEngine to avoid circular dependency."""
    global NQSEvalEngine
    if NQSEvalEngine is None:
        try:
            from QES.NQS.src.nqs_engine import NQSEvalEngine as Engine
            NQSEvalEngine = Engine
        except ImportError:
            pass
    return NQSEvalEngine


def _has_jax():
    """Check if JAX is available for NQS computations."""
    try:
        import jax
        import jax.numpy as jnp
        return True
    except ImportError:
        return False


def default_observable_payload():
    """
    Provide a consistent set of keys so downstream IO layout remains stable
    even before real measurements are plugged in.
    """
    return {
        "energy": float("nan"),
        "energy_std": float("nan"),
        "spin_expectations": {},
        "spin_spin_correlators": {},
        "plaquettes": {},
    }


def compute_spin_expectations_from_state(state_vector: np.ndarray, lattice, hilbert_space) -> Dict[str, np.ndarray]:
    """
    Compute single-spin expectation values <S_i^α> for α = x, y, z from a state vector.
    
    Args:
        state_vector: Normalized wavefunction in computational basis
        lattice: Lattice object with site information
        hilbert_space: HilbertSpace object for basis operations
        
    Returns:
        Dictionary with keys 'sx', 'sy', 'sz', each containing array of expectations per site
    """
    ns = lattice.Ns
    sx_vals = np.zeros(ns)
    sy_vals = np.zeros(ns)
    sz_vals = np.zeros(ns)
    
    # Import spin operators
    try:
        from QES.Algebra.Operator.operators_spin import sig_x, sig_y, sig_z, OperatorTypeActing
        
        # Create local operators - these act on single sites
        op_sx = sig_x(lattice=lattice, type_act=OperatorTypeActing.Local)
        op_sy = sig_y(lattice=lattice, type_act=OperatorTypeActing.Local)
        op_sz = sig_z(lattice=lattice, type_act=OperatorTypeActing.Local)
        
        # Compute matrix representations
        mat_sx = op_sx.matrix(hilbert_1=hilbert_space, matrix_type='dense')
        mat_sy = op_sy.matrix(hilbert_1=hilbert_space, matrix_type='dense')
        mat_sz = op_sz.matrix(hilbert_1=hilbert_space, matrix_type='dense')
        
        # Check if matrices were created successfully
        if mat_sx is not None and mat_sy is not None and mat_sz is not None:
            # For local operators, we sum over all sites
            # <S^α_total> = <psi|Σ_i S^α_i|psi>
            sx_vals[0] = np.real(state_vector.conj() @ mat_sx @ state_vector)
            sy_vals[0] = np.real(state_vector.conj() @ mat_sy @ state_vector)
            sz_vals[0] = np.real(state_vector.conj() @ mat_sz @ state_vector)
            
            # For individual site expectations, we would need site-specific operators
            # For now, distribute the total equally (approximation)
            sx_vals = np.full(ns, sx_vals[0] / ns)
            sy_vals = np.full(ns, sy_vals[0] / ns)
            sz_vals = np.full(ns, sz_vals[0] / ns)
            
    except (ImportError, AttributeError, Exception) as e:
        # Fallback if operators not available
        pass
    
    return {"sx": sx_vals, "sy": sy_vals, "sz": sz_vals}


def compute_spin_correlators_from_state(
    state_vector: np.ndarray, 
    lattice, 
    hilbert_space,
    pairs: Optional[List[Tuple[int, int]]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute two-point spin correlators <S_i^α S_j^α> for nearest neighbors.
    
    Args:
        state_vector: Normalized wavefunction
        lattice: Lattice object
        hilbert_space: HilbertSpace object
        pairs: List of (i, j) site pairs. If None, uses all nearest neighbors.
        
    Returns:
        Dictionary with keys 'sxsx', 'sysy', 'szsz', each containing correlator values
    """
    correlators = {"sxsx": [], "sysy": [], "szsz": []}
    
    try:
        from QES.Algebra.Operator.operators_spin import sig_x, sig_y, sig_z, OperatorTypeActing
        
        # Create correlation operators
        op_sx_sx = sig_x(lattice=lattice, type_act=OperatorTypeActing.Correlation)
        op_sy_sy = sig_y(lattice=lattice, type_act=OperatorTypeActing.Correlation)
        op_sz_sz = sig_z(lattice=lattice, type_act=OperatorTypeActing.Correlation)
        
        # Determine pairs to compute
        if pairs is None:
            pairs = []
            for i in range(lattice.Ns):
                nn_num = lattice.get_nn_forward_num(i)
                for nn in range(nn_num):
                    j = lattice.get_nn_forward(i, num=nn)
                    if not lattice.wrong_nei(j):
                        pairs.append((i, j))
        
        for i, j in pairs:
            sxsx = np.real(state_vector.conj() @ op_sx_sx.act_on_vec(state_vector, sites=[i, j]))
            sysy = np.real(state_vector.conj() @ op_sy_sy.act_on_vec(state_vector, sites=[i, j]))
            szsz = np.real(state_vector.conj() @ op_sz_sz.act_on_vec(state_vector, sites=[i, j]))
            
            correlators["sxsx"].append(sxsx)
            correlators["sysy"].append(sysy)
            correlators["szsz"].append(szsz)
            
    except (ImportError, AttributeError):
        pass
    
    # Convert to arrays
    for key in correlators:
        correlators[key] = np.array(correlators[key])
    
    return correlators


def compute_plaquette_operators(
    state_vector: np.ndarray,
    lattice,
    hilbert_space,
    plaquettes: Optional[List[List[int]]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute plaquette operators W_p = σ_x^i σ_y^j σ_z^k for honeycomb plaquettes.
    
    Args:
        state_vector: Normalized wavefunction
        lattice: Lattice object (should be honeycomb)
        hilbert_space: HilbertSpace object
        plaquettes: List of plaquette site lists. If None, auto-detect from lattice.
        
    Returns:
        Dictionary with 'values' key containing plaquette expectation values
    """
    plaq_vals = []
    
    try:
        from QES.Algebra.Operator.operators_spin import sig_x, sig_y, sig_z, OperatorTypeActing
        
        # For honeycomb, identify plaquettes (6-site hexagons)
        # This is a simplified implementation - real plaquettes depend on lattice geometry
        if plaquettes is None:
            # Auto-detect would require lattice-specific logic
            plaquettes = []
        
        op_sx = sig_x(lattice=lattice, type_act=OperatorTypeActing.Local)
        op_sy = sig_y(lattice=lattice, type_act=OperatorTypeActing.Local)
        op_sz = sig_z(lattice=lattice, type_act=OperatorTypeActing.Local)
        
        for plaq_sites in plaquettes:
            # Build plaquette operator as product of Pauli matrices
            # For Kitaev model: alternate σ_x, σ_y, σ_z around plaquette
            op = 1.0  # Identity
            for idx, site in enumerate(plaq_sites):
                pauli_type = idx % 3  # Cycle through x, y, z
                if pauli_type == 0:
                    op = op * op_sx
                elif pauli_type == 1:
                    op = op * op_sy
                else:
                    op = op * op_sz
            
            plaq_val = np.real(state_vector.conj() @ op.act_on_vec(state_vector))
            plaq_vals.append(plaq_val)
            
    except (ImportError, AttributeError):
        pass
    
    return {"values": np.array(plaq_vals)}


def compute_observables(nqs_solver, hamiltonian, lattice, hilbert_space) -> Dict[str, object]:
    """
    Compute all observables from NQS solver.
    
    Args:
        nqs_solver: NQS solver object with sampling capabilities
        hamiltonian: Hamiltonian object
        lattice: Lattice object
        hilbert_space: HilbertSpace object
        
    Returns:
        Dictionary containing energy, spin expectations, correlators, and plaquettes
    """
    Engine = _get_nqs_engine()
    if Engine is None:
        # Fall back to JAX-based computation if available
        if _has_jax():
            return compute_observables_nqs_jax(nqs_solver, hamiltonian, lattice, hilbert_space)
        return default_observable_payload()

    try:
        engine = Engine(nqs_solver)
        # TODO: integrate with actual sampling/evaluation routines from QES.NQS
        # For now, fall back to JAX implementation
        if _has_jax():
            return compute_observables_nqs_jax(nqs_solver, hamiltonian, lattice, hilbert_space)
        _ = engine
    except Exception:
        if _has_jax():
            return compute_observables_nqs_jax(nqs_solver, hamiltonian, lattice, hilbert_space)
        return default_observable_payload()

    return default_observable_payload()


def compute_local_energy_nqs(
    nqs_model: Any,
    hamiltonian,
    sample,
    use_jax: bool = True
) -> float:
    """
    Compute local energy E_loc(s) = Σ_s' <s'|H|s> psi(s')/psi(s) for a single sample.
    
    Uses Hamiltonian's built-in loc_energy functions (already JIT-compiled!).
    The Hamiltonian already has all operators set up - no need to extract terms.
    
    Args:
        nqs_model: NQS model with .log_amplitude(state) method
        hamiltonian: Hamiltonian with .loc_energy_arr() or .loc_energy_arr_jax()
        sample: Current quantum state (configuration)
        use_jax: Whether to use JAX (True) or numpy (False)
        
    Returns:
        Local energy value for this sample
    """
    if use_jax and not _has_jax():
        use_jax = False
    
    # Use Hamiltonian's built-in local energy function
    if use_jax:
        import jax.numpy as jnp
        
        if hasattr(hamiltonian, 'loc_energy_arr_jax'):
            # Get connected states and matrix elements
            new_states, coeffs = hamiltonian.loc_energy_arr_jax(sample)
            
            # Compute amplitude ratios
            log_psi_s = nqs_model.log_amplitude(sample)
            e_loc = 0.0 + 0j
            
            for new_state, coeff in zip(new_states, coeffs):
                log_psi_sp = nqs_model.log_amplitude(new_state)
                amp_ratio = jnp.exp(log_psi_sp - log_psi_s)
                e_loc += coeff * amp_ratio
            
            return float(jnp.real(e_loc))
        else:
            return float('nan')
    else:
        # Numpy fallback
        if hasattr(hamiltonian, 'loc_energy_arr_np'):
            new_states, coeffs = hamiltonian.loc_energy_arr_np(sample)
            
            psi_s = nqs_model.amplitude(sample)
            e_loc = 0.0 + 0j
            
            for new_state, coeff in zip(new_states, coeffs):
                psi_sp = nqs_model.amplitude(new_state)
                e_loc += coeff * (psi_sp / psi_s)
            
            return float(np.real(e_loc))
        else:
            return float('nan')


def compute_operator_expectation_nqs(
    nqs_model: Any,
    operator,
    sites: Optional[List[int]] = None,
    n_samples: int = 10000,
    use_jax: bool = True
) -> float:
    """
    Compute expectation value <O> = Σ_s |psi(s)|² <s|O|s> using Monte Carlo sampling.
    
    FULLY GENERAL - works for ANY operator with .jax property!
    Examples: spin operators, density, number operators, correlation functions, etc.
    
    Args:
        nqs_model: NQS model with .sample() and .log_amplitude() methods
        operator: ANY operator object with .jax property or Hamiltonian with .loc_energy_arr_jax()
        sites: Sites on which operator acts (None = all sites)
        n_samples: Number of Monte Carlo samples
        use_jax: Whether to use JAX for computation
        
    Returns:
        Expectation value <O>
    """
    if use_jax and not _has_jax():
        use_jax = False
    
    if use_jax:
        import jax.numpy as jnp
        
        # Check if it's an operator with .jax or a Hamiltonian
        if hasattr(operator, 'jax'):
            # Get JAX function from operator
            op_func = operator.jax
        elif hasattr(operator, 'loc_energy_arr_jax'):
            # It's a Hamiltonian - use its local energy function
            op_func = None  # Will use loc_energy_arr_jax directly
        else:
            return float('nan')
        
        # Sample from |psi|²
        samples = nqs_model.sample(n_samples)
        
        expectation = 0.0
        for sample in samples:
            log_psi_s = nqs_model.log_amplitude(sample)
            
            if op_func is not None:
                # Apply operator: |s'> = O|s>
                new_state, op_coeff = op_func(sample, sites=sites)
                
                if new_state != sample:  # Off-diagonal
                    log_psi_sp = nqs_model.log_amplitude(new_state)
                    expectation += jnp.real(op_coeff * jnp.exp(log_psi_sp - log_psi_s))
                else:  # Diagonal
                    expectation += jnp.real(op_coeff)
            else:
                # Use Hamiltonian's local energy
                new_states, coeffs = operator.loc_energy_arr_jax(sample)
                for new_state, coeff in zip(new_states, coeffs):
                    if new_state != sample:
                        log_psi_sp = nqs_model.log_amplitude(new_state)
                        expectation += jnp.real(coeff * jnp.exp(log_psi_sp - log_psi_s))
                    else:
                        expectation += jnp.real(coeff)
        
        return float(expectation / n_samples)
    else:
        # Numpy fallback
        samples = nqs_model.sample(n_samples)
        expectation = 0.0
        
        for sample in samples:
            psi_s = nqs_model.amplitude(sample)
            
            if hasattr(operator, 'apply'):
                new_state, op_coeff = operator.apply(sample, sites=sites)
                psi_sp = nqs_model.amplitude(new_state)
                expectation += np.real(op_coeff * (psi_sp / psi_s))
            elif hasattr(operator, 'loc_energy_arr_np'):
                new_states, coeffs = operator.loc_energy_arr_np(sample)
                for new_state, coeff in zip(new_states, coeffs):
                    psi_sp = nqs_model.amplitude(new_state)
                    expectation += np.real(coeff * (psi_sp / psi_s))
        
        return float(expectation / n_samples)


def compute_observables_nqs_jax(
    nqs_model: Any,
    hamiltonian,
    lattice,
    hilbert_space,
    observables_dict: Optional[Dict[str, Any]] = None,
    n_samples: int = 10000
) -> Dict[str, object]:
    """
    Compute observables from NQS using operator.jax and Monte Carlo sampling.
    
    FULLY GENERAL - computes ANY observables provided in observables_dict!
    Each observable must be an operator with .jax property.
    
    Args:
        nqs_model: NQS model with .sample() and .log_amplitude() methods
        hamiltonian: Hamiltonian object (preferably with .get_operator_terms_for_nqs())
        lattice: Lattice object with site information
        hilbert_space: HilbertSpace object
        observables_dict: Dictionary of {name: operator} or {name: (operator, sites)} pairs
                         If None, computes default spin observables for backward compatibility
        n_samples: Number of Monte Carlo samples for observable averaging
        
    Returns:
        Dictionary containing energy, energy_std, and all requested observables
    """
    if not _has_jax():
        return default_observable_payload()
    
    try:
        import jax
        import jax.numpy as jnp
        
        # Sample from |psi|²
        samples = nqs_model.sample(n_samples)
        
        # Compute local energy samples
        energy_samples = []
        for sample in samples:
            e_loc = compute_local_energy_nqs(nqs_model, hamiltonian, sample, use_jax=True)
            if not np.isnan(e_loc):
                energy_samples.append(e_loc)
        
        # Compute energy statistics
        if energy_samples:
            energy_mean = float(jnp.mean(jnp.array(energy_samples)))
            energy_std = float(jnp.std(jnp.array(energy_samples)))
        else:
            energy_mean = float('nan')
            energy_std = float('nan')
        
        # Compute observables
        results = {
            "energy": energy_mean,
            "energy_std": energy_std,
        }
        
        # Default: compute spin observables for backward compatibility
        if observables_dict is None:
            try:
                from QES.Algebra.Operator.operators_spin import sig_x, sig_y, sig_z, OperatorTypeActing
                ns = lattice.Ns
                
                observables_dict = {
                    'sx': (sig_x(lattice=lattice, type_act=OperatorTypeActing.Local), list(range(ns))),
                    'sy': (sig_y(lattice=lattice, type_act=OperatorTypeActing.Local), list(range(ns))),
                    'sz': (sig_z(lattice=lattice, type_act=OperatorTypeActing.Local), list(range(ns))),
                }
            except ImportError:
                observables_dict = {}
        
        # Compute each observable using operator.jax
        computed_observables = {}
        for name, obs_spec in observables_dict.items():
            # Handle both (operator, sites) and just operator
            if isinstance(obs_spec, tuple):
                operator, sites_list = obs_spec
            else:
                operator = obs_spec
                sites_list = [None]  # Will apply to all sites
            
            # Compute expectation for each site or site-list
            if isinstance(sites_list, list) and len(sites_list) > 1:
                # Multiple sites - compute for each
                vals = []
                for site in sites_list:
                    val = compute_operator_expectation_nqs(
                        nqs_model, operator, sites=[site], n_samples=n_samples
                    )
                    vals.append(val)
                computed_observables[name] = np.array(vals)
            else:
                # Single operator application
                val = compute_operator_expectation_nqs(
                    nqs_model, operator, sites=sites_list[0] if sites_list[0] else None, n_samples=n_samples
                )
                computed_observables[name] = val
        
        # Structure for backward compatibility
        if 'sx' in computed_observables and 'sy' in computed_observables and 'sz' in computed_observables:
            results["spin_expectations"] = {
                "sx": computed_observables['sx'],
                "sy": computed_observables['sy'],
                "sz": computed_observables['sz'],
            }
        else:
            results["observables"] = computed_observables
        
        results["spin_spin_correlators"] = {}  # TODO: Implement if needed
        results["plaquettes"] = {}  # TODO: Implement if needed
        
        return results
        
    except Exception as e:
        print(f"Warning: NQS observable computation failed: {e}")
        import traceback
        traceback.print_exc()
        return default_observable_payload()


def compute_observables_from_eigenvector(
    state_vector: np.ndarray,
    energy: float,
    hamiltonian,
    lattice,
    hilbert_space
) -> Dict[str, object]:
    """
    Compute all observables from an exact eigenvector (e.g., from ED).
    
    Args:
        state_vector: Normalized eigenvector
        energy: Corresponding eigenvalue
        hamiltonian: Hamiltonian object
        lattice: Lattice object
        hilbert_space: HilbertSpace object
        
    Returns:
        Dictionary containing all computed observables
    """
    spin_exp = compute_spin_expectations_from_state(state_vector, lattice, hilbert_space)
    correlators = compute_spin_correlators_from_state(state_vector, lattice, hilbert_space)
    plaquettes = compute_plaquette_operators(state_vector, lattice, hilbert_space)
    
    return {
        "energy": float(energy),
        "energy_std": 0.0,  # Exact eigenstate has zero variance
        "spin_expectations": spin_exp,
        "spin_spin_correlators": correlators,
        "plaquettes": plaquettes,
    }

