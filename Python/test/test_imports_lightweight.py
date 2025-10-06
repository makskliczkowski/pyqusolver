"""
Lightweight import-time health test for QES package.
Ensures that key modules can be imported without double-initialization or errors.
"""

import importlib
import sys

MODULES = [
    "QES",
    "QES.qes_globals",
    "QES.Algebra",
    "QES.Algebra.Operator",
    "QES.Algebra.hilbert",
    "QES.Algebra.hamil", 
    "QES.Algebra.symmetries",
    "QES.general_python.common.flog",
    "QES.general_python.algebra.utils",
]

def main():
    failed = []
    for mod in MODULES:
        try:
            importlib.import_module(mod)
            print(f"[OK] {mod}")
        except Exception as e:
            print(f"[FAIL] {mod}: {e}")
            failed.append(mod)
    if failed:
        sys.exit(1)
    print("All key QES modules imported successfully.")

if __name__ == "__main__":
    main()
