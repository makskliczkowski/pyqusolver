import argparse
import sys
import os

# Add Python/ to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
python_path = os.path.join(root_dir, "Python")

if python_path not in sys.path:
    sys.path.append(python_path)

# Ensure root dir is in path so we can import benchmarks package
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from benchmarks import bench_hamil
    from benchmarks import bench_nqs
except ImportError as e:
    print(f"Error importing benchmarks: {e}")
    # Fallback for running inside benchmarks dir?
    try:
        import bench_hamil
        import bench_nqs
    except ImportError as e2:
        print(f"Critical error: {e2}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="QES Benchmark Suite")
    parser.add_argument("--heavy", action="store_true", help="Run heavy benchmarks (larger systems)")
    parser.add_argument("--filter", type=str, choices=["hamil", "nqs", "all"], default="all", help="Filter benchmarks to run")

    args = parser.parse_args()

    print("Starting QES Benchmarks...")
    print(f"Heavy mode: {args.heavy}")

    if args.filter in ["hamil", "all"]:
        bench_hamil.run_hamil_benchmarks(heavy=args.heavy)

    if args.filter in ["nqs", "all"]:
        bench_nqs.run_nqs_benchmarks(heavy=args.heavy)

    print("\nBenchmarks completed.")

if __name__ == "__main__":
    main()
