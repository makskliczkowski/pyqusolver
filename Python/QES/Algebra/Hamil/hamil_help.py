r'''
Hamiltonian Help Module
=========================
This module provides help texts and documentation snippets for the Hamiltonian class.
'''

hamil_topics = {
    'construction': r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       Hamiltonian: Construction                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Key Parameters:                                                             ║
║    hilbert_space : HilbertSpace - Hilbert space with symmetries              ║
║    lattice       : Lattice      - Lattice geometry (alternative to ns)       ║
║    ns            : int          - Number of sites (if no lattice)            ║
║    is_manybody   : bool         - True for many-body, False for quadratic    ║
║    dtype         : np.dtype     - Data type (float64 or complex128)          ║
║    backend       : str          - 'numpy', 'jax', or 'default'               ║
║                                                                              ║
║  Model-Specific Subclasses (use these instead of base Hamiltonian):          ║
║    HeisenbergKitaev(lattice, K=1.0, J=None, hz=None, ...)                    ║
║    TransverseIsing(lattice, J=1.0, h=1.0, ...)                               ║
║    FreeFermions(ns=N, t=1.0, mu=0.0, ...)                                    ║
║    BCSHamiltonian(ns=N, t=1.0, delta=0.5, ...)                               ║
║                                                                              ║
║  From Configuration:                                                         ║
║    Hamiltonian.from_config(HamiltonianConfig(...))                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
    'operators': r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       Hamiltonian: Adding Operators                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Adding Terms to Hamiltonian:                                                ║
║    .add(operator, multiplier, modifies=False, sites=None)                    ║
║        Add an operator term: multiplier × operator                           ║
║                                                                              ║
║  Operator Collections:                                                       ║
║    .reset_operators()        - Clear all operator terms                      ║
║    ._ops_nmod_nosites        - Operators that don't modify state (global)    ║
║    ._ops_nmod_sites          - Operators that don't modify state (local)     ║
║    ._ops_mod_nosites         - Operators that modify state (global)          ║
║    ._ops_mod_sites           - Operators that modify state (local)           ║
║                                                                              ║
║  Accessing Operators:                                                        ║
║    .operators                - Get the operator module for this Hamiltonian  ║
║    .correlators(pairs, types)- Compute correlation operators                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
    'local_energy': r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     Hamiltonian: Local Energy Functions                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Local Energy (for NQS/VMC):                                                 ║
║    .local_energy(state)      - Compute E_loc = ⟨state|H|state⟩/⟨state|state⟩ ║
║    .fun_int                  - Integer-based local energy function           ║
║    .fun_npy                  - NumPy-based local energy function             ║
║    .fun_jax                  - JAX-based local energy function               ║
║                                                                              ║
║  Instruction Codes (for JIT compilation):                                    ║
║    ._lookup_codes            - Dict mapping operator names to codes          ║
║    ._instr_codes             - Instruction codes for operators               ║
║    ._instr_function          - JIT-compiled instruction function             ║
║    ._instr_max_out           - Maximum output dimension                      ║
║                                                                              ║
║  Setting Up Local Energy:                                                    ║
║    ._set_local_energy_functions()    - Build JIT functions                   ║
║    .setup_instruction_codes()        - Auto-setup based on physics type      ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
    'diagonalization': r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     Hamiltonian: Diagonalization                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Main Method (overrides Operator.diagonalize with extra features):           ║
║    .diagonalize(method='auto', k=None, which='smallest', ...)                ║
║                                                                              ║
║  Hamiltonian-Specific Features:                                              ║
║    - Automatic handling of quadratic Hamiltonians                            ║
║    - Average energy calculation                                              ║
║    - Integration with Hilbert space symmetries                               ║
║                                                                              ║
║  After Diagonalization:                                                      ║
║    .eigenvalues, .eig_val    - Energy eigenvalues                            ║
║    .eigenvectors, .eig_vec   - Energy eigenstates                            ║
║    .ground_state             - Ground state |ψ₀⟩                             ║
║    .ground_energy            - Ground state energy E₀                        ║
║    .spectral_gap             - E₁ - E₀                                       ║
║    .av_en                    - Average energy                                ║
║                                                                              ║
║  Matrix Building:                                                            ║
║    .build_hamiltonian()      - Build the Hamiltonian matrix                  ║
║    .hamil                    - Access the built Hamiltonian matrix           ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
    'quadratic': r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     Hamiltonian: Quadratic Hamiltonians                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Non-Interacting Systems (is_manybody=False):                                ║
║    H = Σᵢⱼ hᵢⱼ c†ᵢ cⱼ + Σᵢⱼ Δᵢⱼ c†ᵢ c†ⱼ + h.c.                              ║
║                                                                              ║
║  Properties:                                                                 ║
║    .is_quadratic             - True if non-interacting                       ║
║    .is_manybody              - True if many-body                             ║
║    ._hamil_sp                - Single-particle Hamiltonian matrix            ║
║    ._delta_sp                - Pairing matrix (for BCS)                      ║
║                                                                              ║
║  Quadratic Diagonalization:                                                  ║
║    - Automatically handled in .diagonalize()                                 ║
║    - Uses single-particle basis transformations                              ║
║    - Much more efficient than many-body (O(N³) vs O(2^N))                    ║
║                                                                              ║
║  Many-Body State Construction:                                               ║
║    .many_body_state(orbitals)- Build many-body state from filled orbitals    ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
    'kspace': r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       Hamiltonian: K-Space Methods                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Transformation to K-Space:                                                  ║
║    .to_kspace()                      - Transform Hamiltonian to k-space      ║
║        Returns: (H_k, kgrid, kgrid_frac)                                     ║
║                                                                              ║
║    .from_kspace(H_k, kgrid)          - Transform back to real space          ║
║        Returns: H_real                                                       ║
║                                                                              ║
║    .transform_operator_to_kspace(O)  - Transform any operator to k-space     ║
║        Returns: (O_k, kgrid, kgrid_frac)                                     ║
║                                                                              ║
║  Requirements:                                                               ║
║    - Lattice must be provided                                                ║
║    - Lattice must support k-space transformations                            ║
║                                                                              ║
║  Use Cases:                                                                  ║
║    - Band structure calculations                                             ║
║    - Momentum-resolved spectral functions                                    ║
║    - Bloch state analysis                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
    'inherited': r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                Hamiltonian: Inherited from Operator/GeneralMatrix            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  From Operator:                                                              ║
║    .apply(state, *args)      - Apply Hamiltonian to state                    ║
║    .matrix(dim, hilbert)     - Generate matrix representation                ║
║    .matvec(v, hilbert)       - Matrix-vector product                         ║
║    .type_acting              - Operator type (usually Global)                ║
║                                                                              ║
║  From GeneralMatrix:                                                         ║
║    Spectral Analysis:                                                        ║
║      .spectral_gap, .spectral_width, .level_spacing()                        ║
║      .participation_ratio(n), .degeneracy(tol)                               ║
║      .level_spacing_ratio()   - For chaos analysis                           ║
║                                                                              ║
║    Matrix Operations:                                                        ║
║      .diag                    - Diagonal elements (auto-selects matrix)      ║
║      .expectation_value(ψ, φ), .overlap(v1, v2)                              ║
║      .trace_matrix(), .frobenius_norm(), .spectral_norm()                    ║
║      .commutator(O), .anticommutator(O)                                      ║
║                                                                              ║
║    Memory & Control:                                                         ║
║      .memory_mb, .memory_gb, .clear()                                        ║
║      .to_sparse(), .to_dense()                                               ║
║                                                                              ║
║  IMPORTANT: Matrix Reference Override                                        ║
║    Hamiltonian overrides _get_matrix_reference() to auto-select:             ║
║      • Many-body (is_manybody=True)  -> uses _hamil (full many-body matrix)   ║
║      • Quadratic (is_manybody=False) -> uses _hamil_sp (single-particle)      ║
║    This ensures diag, trace, norms, etc. use the correct matrix!             ║
║                                                                              ║
║  Use Operator.help() or GeneralMatrix.help() for full inherited methods.     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
}

hamil_overview = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                               Hamiltonian                                    ║
║         Quantum Hamiltonian class for many-body and quadratic systems.       ║
║            Supports diagonalization, local energy, and k-space.              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Inheritance: Hamiltonian -> Operator -> GeneralMatrix -> LinearOperator       ║
║  Model Subclasses: HeisenbergKitaev, TransverseIsing, FreeFermions, ...      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Quick Start:                                                                ║
║    1. Create model:    H = TransverseIsing(ns=10, J=1.0, h=0.5)              ║
║    2. Build matrix:    H.build_hamiltonian()                                 ║
║    3. Diagonalize:     H.diagonalize()                                       ║
║    4. Get results:     E0 = H.ground_energy; psi0 = H.ground_state           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Topics (use .help('topic') for details):                                    ║
║    'construction'    - Building Hamiltonians                                 ║
║    'operators'       - Adding operator terms                                 ║
║    'local_energy'    - Local energy for NQS/VMC                              ║
║    'diagonalization' - Diagonalization methods                               ║
║    'quadratic'       - Non-interacting (quadratic) Hamiltonians              ║
║    'kspace'          - K-space transformations                               ║
║    'inherited'       - Methods from Operator/GeneralMatrix                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""