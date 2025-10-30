"""
Example: Efficient Hamiltonian construction for spin systems using symmetry reduction

This demonstrates how to use the optimized matrix builder to construct
Hamiltonians for spin systems with translation symmetry.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import numpy as np
import numba
from QES.Algebra.hilbert import HilbertSpace
from QES.Algebra.Operator.operator import SymmetryGenerators
from QES.general_python.lattices.square import SquareLattice
from QES.Algebra.Hilbert.matrix_builder import build_operator_matrix

@numba.njit(cache=True)
def sigma_x_operator(state, ns):
    """Transverse field: sum_i sigma_x^i"""
    new_states = np.empty(ns, dtype=np.int64)
    values = np.ones(ns, dtype=np.float64)
    for i in range(ns):
        new_states[i] = state ^ (1 << i)
    return new_states, values

@numba.njit(cache=True)
def sigma_z_operator(state, ns):
    """Longitudinal field: sum_i sigma_z^i"""
    new_states = np.array([state], dtype=np.int64)
    value = 0.0
    for i in range(ns):
        if (state >> i) & 1:
            value += 1.0
        else:
            value -= 1.0
    return new_states, np.array([value], dtype=np.float64)

@numba.njit(cache=True)
def sigma_zz_operator(state, ns):
    """Nearest-neighbor interaction: sum_i sigma_z^i sigma_z^{i+1}"""
    value = 0.0
    for i in range(ns):
        bit_i = (state >> i) & 1
        bit_ip1 = (state >> ((i + 1) % ns)) & 1
        
        si = 1.0 if bit_i else -1.0
        sip1 = 1.0 if bit_ip1 else -1.0
        value += si * sip1
    
    new_states = np.array([state], dtype=np.int64)
    return new_states, np.array([value], dtype=np.float64)

def build_transverse_ising_hamiltonian(ns, J=1.0, h=0.5, k=0):
    """
    Build transverse field Ising Hamiltonian in k-th momentum sector.
    
    H = -J sum_i sigma_z^i sigma_z^{i+1} - h sum_i sigma_x^i
    
    Args:
        ns: Number of spins
        J: Ising coupling
        h: Transverse field strength
        k: Momentum sector
        
    Returns:
        H: Hamiltonian matrix (sparse CSR)
        hs: HilbertSpace object
    """
    lattice = SquareLattice(1, ns)
    hs      = HilbertSpace(
                lattice         =   lattice,
                sym_gen         =   [(SymmetryGenerators.Translation_x, k)],
                gen_mapping     =   True
            )
    
    print(f"   Building H for ns={ns}, k={k}")
    print(f"   Full space: 2^{ns} = {2**ns}")
    print(f"   Reduced space: {hs.dim}")
    print(f"   Reduction factor: {2**ns / hs.dim:.2f}x")
    
    H_zz    = -J * build_operator_matrix(sigma_zz_operator, hilbert_space=hs, sparse=True)
    H_x     = -h * build_operator_matrix(sigma_x_operator, hilbert_space=hs, sparse=True, max_local_changes=ns)
    H       = H_zz + H_x
    
    print(f"   H shape: {H.shape}")
    print(f"   Non-zeros: {H.nnz}/{H.shape[0]**2} ({100*H.nnz/H.shape[0]**2:.1f}%)")
    
    return H, hs

# -------
# Main execution
# -------

def main():
    print("\n" + "="*70)
    print("TRANSVERSE FIELD ISING MODEL - SYMMETRY-REDUCED CONSTRUCTION")
    print("="*70 + "\n")
    
    print("Example 1: Small system (ns=6) - All momentum sectors")
    print("-"*70)
    ns = 6
    J = 1.0
    h = 0.5
    
    ground_states = []
    
    for k in range(ns):
        print(f"\nMomentum sector k={k}:")
        H, hs = build_transverse_ising_hamiltonian(ns, J, h, k)
        
        evals = np.linalg.eigvalsh(H.toarray())
        ground_states.append(evals[0])
        
        print(f"   Ground state energy: {evals[0]:.6f}")
        print(f"   Energy gap: {evals[1] - evals[0]:.6f}")
    
    global_gs = min(ground_states)
    gs_sector = ground_states.index(global_gs)
    print(f"\nGlobal ground state: E_0 = {global_gs:.6f} in sector k={gs_sector}")
    print()
    
    print("Example 2: Larger system (ns=10) - k=0 sector only")
    print("-"*70)
    ns = 10
    
    import time
    t0 = time.time()
    H, hs = build_transverse_ising_hamiltonian(ns, J, h, k=0)
    t1 = time.time()
    
    print(f"   Construction time: {(t1-t0)*1000:.2f} ms")
    
    from scipy.sparse.linalg import eigsh
    t0 = time.time()
    evals, evecs = eigsh(H, k=5, which='SA')
    t1 = time.time()
    
    print(f"   Diagonalization time (5 lowest): {(t1-t0)*1000:.2f} ms")
    print(f"   Ground state energy: {evals[0]:.8f}")
    print(f"   First 5 eigenvalues: {evals}")
    print()
    
    print("Example 3: Performance scaling")
    print("-"*70)
    sizes = [6, 8, 10, 12]
    
    print("   ns | Full dim | Reduced dim |  Reduction  | Time (ms)")
    print("   " + "-"*62)
    
    for ns in sizes:
        t0 = time.time()
        H, hs = build_transverse_ising_hamiltonian(ns, J, h, k=0)
        t1 = time.time()
        
        full_dim = 2**ns
        reduced_dim = hs.dim
        reduction = full_dim / reduced_dim
        time_ms = (t1 - t0) * 1000
        
        print(f"   {ns:2d} |  {full_dim:6d}  |    {reduced_dim:4d}     |   {reduction:5.2f}x   | {time_ms:7.2f}")
    
    print()
    print("="*70)
    print("EFFICIENT HAMILTONIAN CONSTRUCTION DEMONSTRATED (ok) ")
    print("="*70)
    print("\nKey advantages:")
    print("  • Automatic symmetry reduction (up to Ns-fold)")
    print("  • Fast NumPy/Numba implementation")
    print("  • Sparse matrix format for large systems")
    print("  • Clean, modular operator definitions")

if __name__ == "__main__":
    main()
