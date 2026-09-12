from algorithms import lcvx
from parameters import rocket_landing_parameters as p
import plot
import numpy as np
import time

if __name__ == '__main__':
    start_time = time.time()

    N = 40
    rkl = p.Parameters(N)  # Initialize once here
    
    # lcvx.generate_problem3_solver(N)                          # uncomment this if you wish to generate compiled solver
    # lcvx.generate_problem4_solver(N)                          # uncomment this if you wish to generate compiled solver

    # tf, min_d = lcvx.solve_p3(rkl, tf_tol=1)                  # native python solver
    tf, min_d = lcvx.solve_p3_cpg(rkl, tf_tol=3)                # generated compiled solver
    # tf_opt, problem, v_data = lcvx.solve_p4(rkl, min_d, tf)   # native python solver
    tf_opt, problem, v_data = lcvx.solve_p4_cpg(rkl, min_d, tf) # generated compiled solver
    elapsed_time = time.time() - start_time
    
    print(f"\n=== Optimal Result ===")
    print(f"Optimal tf: {tf_opt:.6f}")
    print(f"Time elapsed: {elapsed_time:.2f}s")
    plot.run(tf_opt, problem.x.value, problem.u.value, np.exp(
        problem.z.value), problem.s.value, problem.z.value, v_data)
