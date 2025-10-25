#!/usr/bin/env python3
"""
Quick demonstration of the improved QES import structure.

This script showcases:
1. Lightweight top-level import
2. Lazy subpackage access
3. Module discovery utilities
4. Direct class imports
"""

def main():
    print("=" * 70)
    print("QES Modular Import Demo")
    print("=" * 70)
    print()

    # 1. Lightweight top-level import
    print("1. Lightweight QES import:")
    import QES
    print(f"   Version: {QES.__version__}")
    print(f"   Author:  {QES.__author__}")
    print()

    # 2. Module discovery
    print("2. Module discovery with QES.list_modules():")
    modules = QES.list_modules(include_submodules=False)
    for mod in modules:
        print(f"   {mod['name']:<20} - {mod['description'][:60]}")
    print()

    # 3. Describe specific modules
    print("3. Describe specific modules:")
    for key in ['Algebra', 'NQS', 'Solver']:
        desc = QES.describe_module(key)
        print(f"   {key:<12} : {desc[:60]}")
    print()

    # 4. Lazy subpackage access
    print("4. Lazy subpackage access (via __getattr__):")
    algebra = QES.Algebra
    print(f"   QES.Algebra       -> {algebra}")
    nqs = QES.NQS
    print(f"   QES.NQS           -> {nqs}")
    solver = QES.Solver
    print(f"   QES.Solver        -> {solver}")
    print()

    # 5. Lazy class exports
    print("5. Lazy class exports:")
    hilbert = QES.HilbertSpace
    print(f"   QES.HilbertSpace  -> {hilbert}")
    hamil = QES.Hamiltonian
    print(f"   QES.Hamiltonian   -> {hamil}")
    print()

    # 6. Direct imports
    print("6. Direct imports from subpackages:")
    from QES.Algebra import HilbertSpace, Hamiltonian
    print(f"   HilbertSpace      -> {HilbertSpace}")
    print(f"   Hamiltonian       -> {Hamiltonian}")
    print()

    # 7. Global utilities
    print("7. Global utilities (logger, backend):")
    logger = QES.get_logger()
    backend_mgr = QES.get_backend_manager()
    print(f"   Logger            -> {logger}")
    print(f"   BackendManager    -> {backend_mgr}")
    print()

    print("=" * 70)
    print("All imports work correctly! The package is modular and discoverable.")
    print("=" * 70)

if __name__ == '__main__':
    main()
