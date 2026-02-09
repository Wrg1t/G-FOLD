from src import lcvx


if __name__ == '__main__':
    N = 30
    tf, min_d = lcvx.solve_p3(N, tf_tol=1)
    lcvx.solve_p4(N, min_d, tf)
    # lcvx.generate_problem3_solver(N)
    # lcvx.generate_problem4_solver()
