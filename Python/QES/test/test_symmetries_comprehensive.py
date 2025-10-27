"""
Comprehensive tests for modular symmetry framework with dimension reduction examples.

--------------------------------------------
File        : QES/test/test_symmetries_comprehensive.py
Description : Tests for 1D, 2D, square, and honeycomb lattices with symmetry dimension reduction
Author      : Maksymilian Kliczkowski
Date        : 2025-10-26
--------------------------------------------
"""

import unittest
import numpy as np

try:
    from QES.general_python.lattices.square import SquareLattice
    from QES.general_python.lattices.honeycomb import HoneycombLattice
    from QES.general_python.lattices.lattice import LatticeDirection
except Exception as e:  # pragma: no cover
    SquareLattice       = None
    HoneycombLattice    = None
    LatticeDirection    = None

try:
    from QES.Algebra.Symmetries.translation import TranslationSymmetry
    from QES.Algebra.hilbert import HilbertSpace
    from QES.Algebra.Symmetries.reflection import ReflectionSymmetry
    from QES.Algebra.Symmetries.parity import ParitySymmetry
    from QES.Algebra.Operator.operator import SymmetryGenerators
except Exception as e:  # pragma: no cover
    raise e

# ------------------------------------------------

class TestSymmetriesComprehensive(unittest.TestCase):
    """Comprehensive symmetry tests with dimension reduction demonstrations."""
    
    def setUp(self):
        if SquareLattice is None:
            self.skipTest("Lattice classes not available")
    
    # ------------------------------------------------
    # 1D Examples - Critical for user's reference
    # ------------------------------------------------
    
    def test_1d_chain_translation_k0(self):
        """1D chain with translation at k=0 - matches user's example."""
        print("\n" + "="*80)
        print("1D CHAIN: L=8, Translation k=0")
        print("="*80)
        
        # Create 1D chain
        lat                     = SquareLattice(1, 8)
        sym_gen                 = [(SymmetryGenerators.Translation_x, 0)]  # k=0
        
        # Build Hilbert space
        hs                      = HilbertSpace(8, lattice=lat, sym_gen=sym_gen)
        
        # Print dimension reduction
        full_dim                = 2**8
        reduced_dim             = hs._nh if hasattr(hs, '_nh') else len(hs._reprmap) if hs._reprmap else full_dim
        
        print(f"Full Hilbert space:     {full_dim}")
        print(f"Reduced space (k=0):    {reduced_dim}")
        print(f"Reduction factor:       {full_dim / reduced_dim:.2f}x")
        print(f"Symmetry generators:    {[gen.name for gen, _ in hs._sym_group_sec]}")
        
        self.assertLess(reduced_dim, full_dim, "k=0 should reduce Hilbert space")
        self.assertGreater(reduced_dim, 0, "Reduced space should be non-empty")
    
    def test_1d_chain_translation_reflection_k0(self):
        """1D chain with translation k=0 + reflection - maximum reduction."""
        print("\n" + "="*80)
        print("1D CHAIN: L=8, Translation k=0 + Reflection")
        print("="*80)
        
        lat                     = SquareLattice(1, 8)
        sym_gen                 = [
            (SymmetryGenerators.Translation_x, 0),
            (SymmetryGenerators.Reflection, 1),
        ]
        
        hs                      = HilbertSpace(8, lattice=lat, sym_gen=sym_gen)
        
        full_dim                = 2**8
        reduced_dim             = hs._nh if hasattr(hs, '_nh') else len(hs._reprmap) if hs._reprmap else full_dim
        
        print(f"Full Hilbert space:     {full_dim}")
        print(f"Reduced space (k=0+R):  {reduced_dim}")
        print(f"Reduction factor:       {full_dim / reduced_dim:.2f}x")
        print(f"Symmetry generators:    {[gen.name for gen, _ in hs._sym_group_sec]}")
        
        # Verify both symmetries present
        has_translation         = any(gen == SymmetryGenerators.Translation_x for gen, _ in hs._sym_group_sec)
        has_reflection          = any(gen == SymmetryGenerators.Reflection for gen, _ in hs._sym_group_sec)
        
        self.assertTrue(has_translation, "Translation should be present")
        self.assertTrue(has_reflection, "Reflection should be present at k=0")
        self.assertLess(reduced_dim, full_dim, "Combined symmetries should reduce space")
    
    def test_1d_chain_translation_kpi_reflection_filtered(self):
        """1D chain with translation k=\pi /2 - reflection should be auto-filtered."""
        print("\n" + "="*80)
        print("1D CHAIN: L=8, Translation k=\pi /2 (Reflection auto-filtered)")
        print("="*80)
        
        lat                     = SquareLattice(1, 8)
        sym_gen                 = [
            (SymmetryGenerators.Translation_x, 2),  # k=2\pi ·2/8=\pi /2
            (SymmetryGenerators.Reflection, 1),
        ]
        
        hs                      = HilbertSpace(8, lattice=lat, sym_gen=sym_gen)
        
        full_dim                = 2**8
        reduced_dim             = hs._nh if hasattr(hs, '_nh') else len(hs._reprmap) if hs._reprmap else full_dim
        
        print(f"Full Hilbert space:     {full_dim}")
        print(f"Reduced space (k=\pi /2):  {reduced_dim}")
        print(f"Reduction factor:       {full_dim / reduced_dim:.2f}x")
        print(f"Symmetry generators:    {[gen.name for gen, _ in hs._sym_group_sec]}")
        
        # Verify reflection was filtered
        has_reflection          = any(gen == SymmetryGenerators.Reflection for gen, _ in hs._sym_group_sec)
        
        self.assertFalse(has_reflection, "Reflection should be filtered at k=\pi /2")
    
    def test_1d_chain_parity_on_spins(self):
        """1D chain with Parity Z - spin system."""
        print("\n" + "="*80)
        print("1D CHAIN: L=6, Translation k=0 + Parity Z (spin)")
        print("="*80)
        
        from QES.Algebra.Hilbert.hilbert_local import LocalSpace
        
        lat                     = SquareLattice(1, 6)
        sym_gen                 = [
            (SymmetryGenerators.Translation_x, 0),
            (SymmetryGenerators.ParityZ, 1),
        ]
        
        spin_space              = LocalSpace.default_spin_half()
        hs                      = HilbertSpace(6, lattice=lat, sym_gen=sym_gen, local_space=spin_space)
        
        full_dim                = 2**6
        reduced_dim             = hs._nh if hasattr(hs, '_nh') else len(hs._reprmap) if hs._reprmap else full_dim
        
        print(f"Full Hilbert space:     {full_dim}")
        print(f"Reduced space (T+Pz):   {reduced_dim}")
        print(f"Reduction factor:       {full_dim / reduced_dim:.2f}x")
        print(f"Local space type:       {spin_space.typ.name}")
        print(f"Symmetry generators:    {[gen.name for gen, _ in hs._sym_group_sec]}")
        
        has_parity              = any(gen == SymmetryGenerators.ParityZ for gen, _ in hs._sym_group_sec)
        self.assertTrue(has_parity, "Parity Z should be present for spin systems")
    
    # ------------------------------------------------
    # 2D Square Lattice Examples
    # ------------------------------------------------
    
    def test_2d_square_translation_x(self):
        """2D square lattice with translation in x-direction."""
        print("\n" + "="*80)
        print("2D SQUARE LATTICE: 4x4, Translation_x k=0")
        print("="*80)
        
        lat                     = SquareLattice(2, 4, 4)
        sym_gen                 = [(SymmetryGenerators.Translation_x, 0)]
        
        hs                      = HilbertSpace(16, lattice=lat, sym_gen=sym_gen)
        
        full_dim                = 2**16
        reduced_dim             = hs._nh if hasattr(hs, '_nh') else len(hs._reprmap) if hs._reprmap else full_dim
        
        print(f"Full Hilbert space:     {full_dim}")
        print(f"Reduced space (T_x):    {reduced_dim}")
        print(f"Reduction factor:       {full_dim / reduced_dim:.2f}x")
        print(f"Lattice dimensions:     {lat._lx}x{lat._ly}")
        
        self.assertLess(reduced_dim, full_dim, "Translation_x should reduce space")
    
    def test_2d_square_translation_xy(self):
        """2D square lattice with translations in both x and y directions."""
        print("\n" + "="*80)
        print("2D SQUARE LATTICE: 3x3, Translation_x k=0 + Translation_y k=0")
        print("="*80)
        
        lat                     = SquareLattice(2, 3, 3)
        sym_gen                 = [
            (SymmetryGenerators.Translation_x, 0),
            (SymmetryGenerators.Translation_y, 0),
        ]
        
        hs                      = HilbertSpace(9, lattice=lat, sym_gen=sym_gen)
        
        full_dim                = 2**9
        reduced_dim             = hs._nh if hasattr(hs, '_nh') else len(hs._reprmap) if hs._reprmap else full_dim
        
        print(f"Full Hilbert space:     {full_dim}")
        print(f"Reduced space (T_x+T_y):{reduced_dim}")
        print(f"Reduction factor:       {full_dim / reduced_dim:.2f}x")
        print(f"Lattice dimensions:     {lat._lx}x{lat._ly}")
        
        has_tx                  = any(gen == SymmetryGenerators.Translation_x for gen, _ in hs._sym_group_sec)
        has_ty                  = any(gen == SymmetryGenerators.Translation_y for gen, _ in hs._sym_group_sec)
        
        self.assertTrue(has_tx, "Translation_x should be present")
        self.assertTrue(has_ty, "Translation_y should be present")
    
    # ------------------------------------------------
    # Honeycomb Lattice Examples
    # ------------------------------------------------
    
    def test_honeycomb_translation(self):
        """Honeycomb lattice with translation symmetry."""
        if HoneycombLattice is None:
            self.skipTest("HoneycombLattice not available")
        
        print("\n" + "="*80)
        print("HONEYCOMB LATTICE: 3x3, Translation_x k=0")
        print("="*80)
        
        lat                     = HoneycombLattice(dim=2, lx=3, ly=3)
        ns                      = getattr(lat, 'Ns', None) or getattr(lat, 'ns', None)
        sym_gen                 = [(SymmetryGenerators.Translation_x, 0)]
        
        hs                      = HilbertSpace(ns, lattice=lat, sym_gen=sym_gen)
        
        full_dim                = 2**ns
        reduced_dim             = hs._nh if hasattr(hs, '_nh') else len(hs._reprmap) if hs._reprmap else full_dim
        
        print(f"Full Hilbert space:     {full_dim}")
        print(f"Reduced space (T_x):    {reduced_dim}")
        print(f"Reduction factor:       {full_dim / reduced_dim:.2f}x")
        print(f"Number of sites:        {ns}")
        print(f"Lattice type:           Honeycomb")
        
        self.assertLess(reduced_dim, full_dim, "Translation should reduce honeycomb space")
    
    # ------------------------------------------------
    # Local space validation tests
    # ------------------------------------------------
    
    def test_parity_fermion_rejection(self):
        """Verify Parity is rejected for fermion systems."""
        from QES.Algebra.Hilbert.hilbert_local import LocalSpace
        
        lat                     = SquareLattice(1, 4)
        sym_gen                 = [(SymmetryGenerators.ParityX, 1)]
        fermion_space           = LocalSpace.default_fermion_spinless()
        
        hs                      = HilbertSpace(4, lattice=lat, sym_gen=sym_gen, local_space=fermion_space)
        
        has_parity              = any(gen == SymmetryGenerators.ParityX for gen, _ in hs._sym_group_sec)
        
        self.assertFalse(has_parity, "ParityX should be rejected for fermions")


if __name__ == "__main__":
    unittest.main(verbosity=2)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EOF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
