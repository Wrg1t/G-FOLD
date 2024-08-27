from src import lcvx


if __name__ == '__main__':
    DT = 5  # time interval
    N, min_d = lcvx.solve_p3(DT)
    lcvx.solve_p4(DT, N, min_d)
    # lcvx.generate_problem3_solver()
    # lcvx.generate_problem4_solver()
