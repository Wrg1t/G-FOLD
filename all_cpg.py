import sys
from algorithms import lcvx



if len(sys.argv) < 2:
    print("Usage: python all_cpg.py <N>")
    print("Example: python all_cpg.py 30")
    sys.exit(1)

try:
    N = int(sys.argv[1])
except ValueError:
    print(f"Error: N must be an integer, got '{sys.argv[1]}'")
    sys.exit(1)

print(f"Generating CPG solvers for N={N}...")

# Generate Problem 3 solver
print(f"Generating Problem 3 solver (lcvxP3_N{N}_cpg)...")
lcvx.generate_problem3_solver(N)
print(f"✓ Problem 3 solver generated successfully")

# Generate Problem 4 solver
print(f"Generating Problem 4 solver (lcvxP4_N{N}_cpg)...")
lcvx.generate_problem4_solver(N)
print(f"✓ Problem 4 solver generated successfully")

print(f"\nAll solvers for N={N} have been generated.")

