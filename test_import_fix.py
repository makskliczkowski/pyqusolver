
try:
    from QES.Algebra.globals import violates_global_syms
    print("Successfully imported violates_global_syms from QES.Algebra.globals")
except ImportError as e:
    print(f"Failed to import: {e}")

try:
    from QES.Algebra.Symmetries.jit.symmetry_container_jit import violates_global_syms as vgs
    print("Successfully imported QES.Algebra.Symmetries.jit.symmetry_container_jit")
except ImportError as e:
    print(f"Failed to import jit: {e}")
