#!/usr/bin/env python3
"""
Example: Using the lazy operator loader via Hamiltonian and HilbertSpace.

This demonstrates how to access operators without explicit imports:
1. Via Hamiltonian.operators
2. Via HilbertSpace.operators
3. Getting matrix representations
4. Using operators for correlation functions in k-space
"""
import numpy as np
import sys
sys.path.insert(0, "/Users/makskliczkowski/Codes/pyqusolver/Python")

from QES.general_python.lattices.honeycomb import HoneycombLattice  
from QES.Algebra.Model.Noninteracting.Conserving.Majorana.kitaev_gamma_majorana import KitaevGammaMajorana
from QES.general_python.lattices.tools.lattice_kspace import kspace_from_realspace
from QES.Algebra.hilbert import HilbertSpace

print("="*70)
print("Example: Lazy Operator Loading")
print("="*70)

# Create a small lattice
lat = HoneycombLattice(dim=2, lx=2, ly=2, bc='pbc')
Ns = lat.Ns

print(f"\nLattice: {lat._lx}x{lat._ly}, Ns={Ns}")

# Example 1: Access operators via Hamiltonian
print("\n" + "="*70)
print("Example 1: Access operators via Hamiltonian")
print("="*70)

model = KitaevGammaMajorana(lat, k_x=1.0, k_y=1.0, k_z=1.0, 
                            gamma_x=None, gamma_y=None, gamma_z=None, 
                            p_flip=0.0, dtype=complex)
model.build()

# Access operators without importing
ops = model.operators

print(f"\nOperator module loaded: {type(ops).__name__}")
print(f"Available operators: {dir(ops)}")

# For Majorana/BdG systems, we can still access the underlying operator framework
# Let's show this works with a simple spin system instead
print("\n" + "="*70)
print("Example 2: Spin operators via HilbertSpace")
print("="*70)

# Create a spin-1/2 Hilbert space
hilbert = HilbertSpace(ns=4, local_space='spin-1/2', is_manybody=True)
print(f"\nHilbert space: {hilbert}")

# Get operators
ops = hilbert.operators
print(f"\nOperator module type: {type(ops).__name__}")

# Create sigma_x operator on sites [0, 1]
sig_x = ops.sig_x(ns=4, sites=[0, 1])
print(f"\nCreated σ_x operator: {sig_x.name}")
print(f"  Acts on sites: {sig_x.sites}")
print(f"  Modifies state: {sig_x.modifies}")

# Get matrix representation
sig_x_mat = sig_x.matrix
print(f"  Matrix shape: {sig_x_mat.shape}")
print(f"  Matrix non-zeros: {np.count_nonzero(sig_x_mat)}")

# Create other operators
sig_z = ops.sig_z(ns=4, sites=[0])
print(f"\nCreated σ_z operator: {sig_z.name}")
print(f"  Acts on sites: {sig_z.sites}")

sig_z_mat = sig_z.matrix
print(f"  Matrix shape: {sig_z_mat.shape}")
print(f"  Matrix diagonal: {np.diag(sig_z_mat)[:8]}")

# Example 3: Using operators for correlation functions
print("\n" + "="*70)
print("Example 3: Correlation functions with k-space")
print("="*70)

# Back to our Kitaev model
model = KitaevGammaMajorana(lat, k_x=1.0, k_y=1.0, k_z=1.0, 
                            gamma_x=None, gamma_y=None, gamma_z=None, 
                            p_flip=0.0, dtype=complex)
model.build()
H_real = model._hamiltonian_quadratic().toarray()

# Transform to k-space and get W matrix
Hk, kgrid, kgrid_frac, W = kspace_from_realspace(lat, H_real, return_transform=True)

print(f"\nK-space transformation completed:")
print(f"  Hk shape: {Hk.shape}")
print(f"  W shape: {W.shape}")
print(f"  kgrid shape: {kgrid.shape}")

# We can use W to transform any operator to k-space
# For Majorana systems, operators are quadratic forms
print(f"\nBloch unitary W can transform operators: O_k = W† @ O @ W")

# Example 4: Show help
print("\n" + "="*70)
print("Example 4: Getting help on available operators")
print("="*70)

hilbert.operators.help()

# Example 5: Chaining operators
print("\n" + "="*70)
print("Example 5: Operator composition")
print("="*70)

# Create multiple operators and combine them
sig_x_0 = ops.sig_x(ns=4, sites=[0])
sig_x_1 = ops.sig_x(ns=4, sites=[1])

# Operators can be composed
combined = sig_x_0 * sig_x_1
print(f"\nCombined operator σ_x(0) * σ_x(1):")
print(f"  Name: {combined.name}")

# Example 6: Fermion operators (if available)
print("\n" + "="*70)
print("Example 6: Fermion operators")
print("="*70)

try:
    from QES.Algebra.Hilbert.hilbert_local import LocalSpaceTypes
    
    # Create fermion Hilbert space
    hilbert_fermion = HilbertSpace(ns=4, local_space=LocalSpaceTypes.SPINLESS_FERMIONS, 
                                  is_manybody=True)
    print(f"\nFermion Hilbert space: {hilbert_fermion}")
    
    # Get fermion operators
    fops = hilbert_fermion.operators
    
    # Create c_dag operator
    c_dag = fops.c_dag(ns=4, sites=[0])
    print(f"\nCreated c† operator: {c_dag.name}")
    print(f"  Acts on sites: {c_dag.sites}")
    
    # Number operator
    n_op = fops.n(ns=4, sites=[0])
    print(f"\nCreated n operator: {n_op.name}")
    print(f"  Acts on sites: {n_op.sites}")
    
except Exception as e:
    print(f"\nFermion operators not available: {e}")

print("\n" + "="*70)
print("✓ All examples completed!")
print("="*70)
print("\nKey takeaways:")
print("  1. Access operators via: hamil.operators or hilbert.operators")
print("  2. No need to import operator modules explicitly")
print("  3. Operators are lazy-loaded on first access")
print("  4. Get matrix: operator.matrix")
print("  5. Transform to k-space with W: O_k = W† @ O @ W")
print("  6. Use operators.help() to see available operators")
