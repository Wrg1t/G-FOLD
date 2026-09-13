import argparse
import time

import numpy as np

from algorithms import lcvx
from parameters import rocket_landing_parameters as p
import plot

# Solver-specific default tolerance for the golden-section search on flight time.
# The generated solver is fast enough that a looser tolerance is a good trade.
DEFAULT_TF_TOL = {'clarabel': 1.0, 'cpg': 3.0}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description='Fuel-optimal powered descent guidance (G-FOLD)')
    parser.add_argument(
        '-f', '--params',
        metavar='JSON',
        type=str,
        default=None,
        help='path to JSON file containing vessel parameters '
             '(default: parameters/vessel_parameters_mars.json)')
    parser.add_argument(
        '-n',
        metavar='Positive Integer',
        type=int,
        default=40,
        dest='n',
        help='number of intervals (default: 40)')
    parser.add_argument(
        '--solver',
        choices=('clarabel', 'cpg'),
        default='clarabel',
        help="'clarabel' uses the native cvxpy problem through CLARABEL; "
             "'cpg' uses a generated solver from python all_cpg.py <N> "
             "(default: clarabel)")
    parser.add_argument(
        '--tf-tol',
        type=float,
        default=None,
        metavar='SECONDS',
        help='golden-section tolerance on flight time '
             '(default: 1.0 for clarabel, 3.0 for cpg)')

    args = parser.parse_args(argv)

    if args.n <= 0:
        parser.error("Number of intervals (N) must be a positive integer.")
    if args.tf_tol is not None and args.tf_tol <= 0:
        parser.error("--tf-tol must be positive.")

    return args


def main(argv=None):
    args = parse_arguments(argv)
    tf_tol = args.tf_tol
    if tf_tol is None:
        tf_tol = DEFAULT_TF_TOL[args.solver]

    start_time = time.time()

    rkl = p.Parameters(args.n, param_path=args.params)  # Initialize once here

    if args.solver == 'cpg':
        tf, min_d = lcvx.solve_p3_cpg(rkl, tf_tol=tf_tol)
        tf_opt, problem, v_data = lcvx.solve_p4_cpg(rkl, min_d, tf)
    else:
        tf, min_d = lcvx.solve_p3(rkl, tf_tol=tf_tol)
        tf_opt, problem, v_data = lcvx.solve_p4(rkl, min_d, tf)

    elapsed_time = time.time() - start_time

    print(f"\n=== Optimal Result ===")
    print(f"Optimal tf: {tf_opt:.6f}")
    print(f"Time elapsed: {elapsed_time:.2f}s")
    plot.run(tf_opt, problem.x.value, problem.u.value, np.exp(
        problem.z.value), problem.s.value, problem.z.value, v_data)

    return tf_opt


if __name__ == '__main__':
    main()