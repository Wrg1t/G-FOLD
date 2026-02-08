import cvxpy as cp
import numpy as np
from cvxpygen import cpg
from parameters import rocket_landing_parameters as p
from src import lcvx_problem_definition
from src import plot


def plot_cost_time_chart():
    DT = 5  # time interval
    rkl = p.Parameters(DT)

    costs = {}
    for N in range(10, 80):
        try:
            params = rkl.get_data(N)
            lcvx = lcvx_problem_definition.RocketLanding(N, params)
            problem = lcvx.problem3()
            problem.solve(solver='CLARABEL', verbose=False)
            cost = np.linalg.norm(lcvx.x.value[0:3, N - 1] - lcvx.rf.value)
            print(f"N = {N}, cost = {cost}")
            costs[N * DT] = cost
        except:
            print(f"Skipping N = {N} due to a SolverError.")
            continue

    from xyplot import xyplot
    xyplot(costs)

def solve_p3(DT):
    rkl = p.Parameters(DT)

    N_lower_bound = round(rkl.time_lower_bound / DT) + 1
    N_upper_bound = round(rkl.time_upper_bound / DT) + 1

    max_tries = 0
    min_cost = float('inf')
    for N in range(N_lower_bound, N_upper_bound + 1):
        params = rkl.get_data(N)
        lcvx = lcvx_problem_definition.RocketLanding(N, params)
        problem = lcvx.problem3()

        try:
            problem.solve(solver='CLARABEL', verbose=False)
        except:
            print(f"Skipping N = {N} due to a SolverError.")
            continue

        if not problem.status == cp.OPTIMAL:
            print(f"N = {N} is infeasible.")
            if min_cost != float('inf'):
                max_tries += 1
        else:
            cost = np.linalg.norm(lcvx.x.value[0:3, N - 1] - lcvx.rf.value)
            print(f"N = {N}, cost = {cost}")

            if cost < min_cost:
                min_cost = cost
                best_N = N

        if max_tries >= 3:
            break
        elif max_tries < 3 and problem.status == cp.OPTIMAL:
            max_tries = 0

    return best_N, min_cost

def solve_p4(DT, N, min_d):
    rkl = p.Parameters(DT)
    params = rkl.get_data(N)
    lcvx = lcvx_problem_definition.RocketLanding(N, params)
    problem = lcvx.problem4(min_d)
    problem.solve(solver='CLARABEL', verbose=False)

    print(f"Time of flight: {(N - 1) * DT}s")
    plot.run((N - 1) * DT, lcvx.x.value, lcvx.u.value, np.exp(
        lcvx.z.value), lcvx.s.value, lcvx.z.value, rkl.vessel_data)

# codegen INOP, can't handle N dynamically!!!
def generate_problem3_solver():
    DT = 1  # time interval
    N = 80
    rkl = p.Parameters(DT)
    params = rkl.get_data(N)
    lcvx = lcvx_problem_definition.RocketLanding(N, params)
    problem = lcvx.problem3()

    cpg.generate_code(problem, code_dir='lcvxP3', solver='CLARABEL')

def generate_problem4_solver():
    DT = 1  # time interval
    N = 80
    rkl = p.Parameters(DT)
    params = rkl.get_data(N)
    lcvx = lcvx_problem_definition.RocketLanding(N, params)
    problem = lcvx.problem4(1)
    cpg.generate_code(problem, code_dir='lcvxP4', solver='CLARABEL')
