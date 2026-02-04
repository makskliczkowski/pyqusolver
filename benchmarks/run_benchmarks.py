import sys
import os
import argparse
import importlib

# Adjust path to include Python and Python/QES
# Assumes this script is in <repo_root>/benchmarks/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "Python"))
sys.path.append(os.path.join(REPO_ROOT, "Python", "QES"))

def main():
    parser = argparse.ArgumentParser(description="Run QES Benchmarks")
    parser.add_argument("--heavy", action="store_true", help="Run heavy benchmarks (larger systems, more samples)")
    parser.add_argument("--filter", type=str, help="Filter benchmarks by module name (e.g. 'vmc')")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats per benchmark (default: 3)")
    args = parser.parse_args()

    # List of benchmark modules to run
    # These must correspond to files in the benchmarks/ directory
    modules = [
        "benchmarks.hamil_benchmark",
        "benchmarks.vmc_benchmark",
        "benchmarks.nqs_benchmark",
    ]

    print("=" * 60)
    print(f"Running QES Benchmarks (Heavy: {args.heavy}, Repeats: {args.repeats})")
    print("=" * 60)

    for mod_name in modules:
        # Filter logic
        if args.filter and args.filter not in mod_name:
            continue

        try:
            # Dynamic import
            mod = importlib.import_module(mod_name)

            # Check for 'run' entry point
            if hasattr(mod, "run"):
                print(f"\n>> Module: {mod_name}")
                print("-" * 30)
                mod.run(heavy=args.heavy, repeats=args.repeats)
            else:
                print(f"\nSkipping {mod_name}: No 'run' function found.")

        except ImportError as e:
            # Gracefully handle missing dependencies (e.g. if JAX is missing for NQS)
            print(f"\nSkipping {mod_name}: {e}")
        except Exception as e:
            print(f"\nError running {mod_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Benchmarks completed.")
    print("=" * 60)

if __name__ == "__main__":
    main()
