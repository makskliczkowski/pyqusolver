#!/usr/bin/env python3
"""
Integration test for QES modular design improvements.

This test validates:
1. Lightweight top-level import
2. Lazy subpackage loading
3. Module discovery APIs
4. Convenience class exports
5. Global singleton access
6. Backward-compatible imports
"""

import sys
import importlib

# -------------------------------------------------------------------

def test_lightweight_import():
    """Verify that importing QES doesn't pull in heavy dependencies."""
    print("Test 1: Lightweight top-level import...")
    
    # Clear any previously imported QES modules
    for key in list(sys.modules.keys()):
        if key.startswith('QES'):
            del sys.modules[key]
    
    import QES
    
    # At this point, heavy modules like Algebra.hilbert should NOT be loaded
    assert 'QES.Algebra.hilbert' not in sys.modules, "hilbert imported too early"
    assert 'QES.Algebra.hamil' not in sys.modules, "hamil imported too early"
    
    print("  v Top-level import is lightweight")

# -------------------------------------------------------------------

def test_lazy_subpackages():
    """Verify that subpackages are loaded lazily."""
    print("Test 2: Lazy subpackage loading...")
    
    import QES
    
    # Access Algebra via lazy __getattr__
    algebra = QES.Algebra
    assert 'QES.Algebra' in sys.modules, "Algebra not loaded after access"
    assert algebra is not None
    
    print("  v Lazy subpackage loading works")

# -------------------------------------------------------------------

def test_module_discovery():
    """Test the registry APIs."""
    print("Test 3: Module discovery...")
    
    import QES
    
    # list_modules should return a list of dicts
    modules = QES.list_modules(include_submodules=False)
    assert isinstance(modules, list), "list_modules should return a list"
    assert len(modules) > 0, "Should have at least one module"
    assert all('name' in m and 'description' in m for m in modules), "Each entry needs name and description"
    
    # describe_module should return a string
    desc = QES.describe_module('Algebra')
    assert isinstance(desc, str), "describe_module should return a string"
    assert len(desc) > 0, "Description should not be empty"
    
    print(f"  v Found {len(modules)} modules with descriptions")

# -------------------------------------------------------------------

def test_convenience_exports():
    """Test that HilbertSpace and Hamiltonian can be imported from QES."""
    print("Test 4: Convenience class exports...")
    
    import QES
    
    # Lazy class access via __getattr__
    hs_class    = QES.HilbertSpace
    ham_class   = QES.Hamiltonian
    
    assert hs_class.__name__ == 'HilbertSpace'
    assert ham_class.__name__ == 'Hamiltonian'
    
    # Also verify they're the same as the canonical location
    from QES.Algebra.hilbert import HilbertSpace
    from QES.Algebra.hamil import Hamiltonian
    
    assert hs_class is HilbertSpace
    assert ham_class is Hamiltonian
    
    print("  v Convenience exports work and match canonical imports")

# -------------------------------------------------------------------

def test_global_singletons():
    """Test that global singletons are accessible."""
    print("Test 5: Global singleton access...")
    
    import QES
    
    logger      = QES.get_logger()
    backend_mgr = QES.get_backend_manager()
    
    assert logger is not None
    assert backend_mgr is not None
    
    # Verify singleton property
    logger2 = QES.get_logger()
    assert logger is logger2, "Logger should be a singleton"
    
    print("  v Global singletons accessible and unique")

# -------------------------------------------------------------------

def test_backward_compatibility():
    """Test that old import patterns still work."""
    print("Test 6: Backward compatibility...")
    
    # These should all work
    from QES.Algebra import HilbertSpace, Hamiltonian
    from QES.Algebra.Operator import operators_spin
    from QES.qes_globals import get_logger, get_backend_manager
    
    assert HilbertSpace is not None
    assert Hamiltonian is not None
    assert operators_spin is not None
    
    print("  v Backward-compatible imports work")

# -------------------------------------------------------------------

def main():
    """Run all integration tests."""
    print("=" * 70)
    print("QES Modular Design Integration Tests")
    print("=" * 70)
    print()
    
    tests = [
        test_lightweight_import,
        test_lazy_subpackages,
        test_module_discovery,
        test_convenience_exports,
        test_global_singletons,
        test_backward_compatibility,
    ]
    
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"  (x) FAILED: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"  (x) ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    print()
    print("=" * 70)
    print("(v) All integration tests passed!")
    print("=" * 70)

# -------------------------------------------------------------------

if __name__ == '__main__':
    main()

# -------------------------------------------------------------------