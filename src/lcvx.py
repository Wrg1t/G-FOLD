import cvxpy as cp
import numpy as np
from cvxpygen import cpg
from parameters import rocket_landing_parameters as p
from src import lcvx_problem_definition as pdef
from src import plot


def plot_cost_time_chart():
    DT = 5  # time interval
    rkl = p.Parameters(DT)

    costs = {}
    for N in range(10, 80):
        try:
            params = rkl.get_data(N)
            lcvx = pdef.RocketLanding(N, params)
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

def solve_p3(N, tf_tol=10):
    rkl = p.Parameters(N)

    phi = (np.sqrt(5) - 1) * 0.5  # golden ratio
    
    tf_min, tf_max = rkl.time_lower_bound, rkl.time_upper_bound

    tf_left = tf_max - phi * (tf_max - tf_min)
    tf_right = tf_min + phi * (tf_max - tf_min)
    

    left_params = rkl.get_data(tf_left)
    right_params = rkl.get_data(tf_right)
    
    def solve_lcvx(params):
        lcvx = pdef.RocketLanding(N, params)
        problem = lcvx.problem3()
        
        problem.solve(solver='CLARABEL', verbose=False)
        
        if problem.status == cp.OPTIMAL:
            return problem.value
        else:
            return float('inf')

    obj_left = solve_lcvx(left_params)
    obj_right = solve_lcvx(right_params)
    
    max_iters = 100
    iteration = 0 
    while (tf_max - tf_min) > tf_tol and iteration < max_iters:    
        print(f"Iter {iteration}: [{tf_min:.4f}, {tf_max:.4f}] | "
          f"tf_left={tf_left:.4f} (obj={obj_left:.6f}), tf_right={tf_right:.4f} (obj={obj_right:.6f})")

        if obj_left < obj_right:
            tf_max = tf_right
            tf_right = tf_left
            obj_right = obj_left
            
            tf_left = tf_max - phi * (tf_max - tf_min)
            left_params = rkl.get_data(tf_left)
            obj_left = solve_lcvx(left_params)

        else:
            tf_min = tf_left
            tf_left = tf_right
            obj_left = obj_right
            
            tf_right = tf_min + phi * (tf_max - tf_min)
            right_params = rkl.get_data(tf_right)
            obj_right = solve_lcvx(right_params)

        iteration += 1
    
    tf_opt = (tf_max + tf_min) / 2
    opt_params = rkl.get_data(tf_opt)
    obj_opt = solve_lcvx(opt_params)
    
    return tf_opt, obj_opt

def solve_p4(DT, N, min_d):
    rkl = p.Parameters(DT)
    params = rkl.get_data(N)
    lcvx = pdef.RocketLanding(N, params)
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
    lcvx = pdef.RocketLanding(N, params)
    problem = lcvx.problem3()

    cpg.generate_code(problem, code_dir='lcvxP3', solver='CLARABEL')

def generate_problem4_solver():
    DT = 1  # time interval
    N = 80
    rkl = p.Parameters(DT)
    params = rkl.get_data(N)
    lcvx = pdef.RocketLanding(N, params)
    problem = lcvx.problem4(1)
    cpg.generate_code(problem, code_dir='lcvxP4', solver='CLARABEL')
