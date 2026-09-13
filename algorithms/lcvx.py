import cvxpy as cp
import numpy as np
from cvxpygen import cpg
from algorithms import lcvx_problem_definition as pdef


def golden_section_search(params, problem_builder, solve_fn, tf_tol=10, max_iters=100):
    phi = (np.sqrt(5) - 1) * 0.5  # golden ratio
    
    tf_min, tf_max = params.time_lower_bound, params.time_upper_bound

    tf_left = tf_max - phi * (tf_max - tf_min)
    tf_right = tf_min + phi * (tf_max - tf_min)

    left_params = params.get_data(tf_left)
    right_params = params.get_data(tf_right)
    
    obj_left = solve_fn(left_params)
    obj_right = solve_fn(right_params)
    
    iteration = 0 
    while (tf_max - tf_min) > tf_tol and iteration < max_iters:    
        print(f"Iter {iteration}: [{tf_min:.4f}, {tf_max:.4f}] | "
          f"tf_left={tf_left:.4f} (obj={obj_left:.6f}), tf_right={tf_right:.4f} (obj={obj_right:.6f})")

        if obj_left < obj_right:
            tf_max = tf_right
            tf_right = tf_left
            obj_right = obj_left
            
            tf_left = tf_max - phi * (tf_max - tf_min)
            left_params = params.get_data(tf_left)
            obj_left = solve_fn(left_params)

        else:
            tf_min = tf_left
            tf_left = tf_right
            obj_left = obj_right
            
            tf_right = tf_min + phi * (tf_max - tf_min)
            right_params = params.get_data(tf_right)
            obj_right = solve_fn(right_params)

        iteration += 1
    
    tf_opt = (tf_max + tf_min) / 2
    opt_params = params.get_data(tf_opt)
    obj_opt = solve_fn(opt_params)
    
    return tf_opt, obj_opt


def solve_p3(params, tf_tol=10):
    N = params.N
    
    def problem_builder(param_data):
        lcvx = pdef.RocketLanding(N, param_data)
        problem = lcvx.problem3()
        return problem
    
    def solve_lcvx(param_data):
        problem = problem_builder(param_data)
        problem.solve(solver='CLARABEL', verbose=False)
        if problem.status == cp.OPTIMAL:
            return problem.value
        else:
            return float('inf')
    
    tf_opt, obj_opt = golden_section_search(params, problem_builder, solve_lcvx, tf_tol)
    return tf_opt, obj_opt


def solve_p4(params, min_d, tf_opt):
    N = params.N
    opt_params = params.get_data(tf_opt)    
    lcvx = pdef.RocketLanding(N, opt_params)
    problem = lcvx.problem4(min_d)
    problem.solve(solver='CLARABEL', verbose=False)

    return (tf_opt, lcvx, params.vessel_data)


def solve_p3_cpg(params, tf_tol=10):
    N = params.N
    
    def problem_builder(param_data):
        lcvx = pdef.RocketLanding(N, param_data)
        problem = lcvx.problem3()
        return problem
    
    def solve_lcvx(param_data):
        problem = problem_builder(param_data)
        
        import importlib
        module_name = f"lcvxP3_N{N}_cpg.cpg_solver"
        
        try:
            lcvxP3_solver = importlib.import_module(module_name)
            cpg_solve = lcvxP3_solver.cpg_solve
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"Solver module '{module_name}' not found. "
                f"Please generate the solver by running: python all_cpg.py {N}"
            )

        problem.register_solve('CPG', cpg_solve)
        problem.solve(method='CPG', verbose=False)
        if not np.isnan(problem.value):
            return problem.value
        else:
            return float('inf')
        
    import sys
    # after solving, unload the module to avoid conflicts
    if f"lcvxP3_N{N}_cpg" in sys.modules:
        del sys.modules[f"lcvxP3_N{N}_cpg"]
    if f"lcvxP3_N{N}_cpg.cpg_solver" in sys.modules:
        del sys.modules[f"lcvxP3_N{N}_cpg.cpg_solver"]
    
    tf_opt, obj_opt = golden_section_search(params, problem_builder, solve_lcvx, tf_tol)
    return tf_opt, obj_opt


def solve_p4_cpg(params, min_d, tf_opt):
    N = params.N
    opt_params = params.get_data(tf_opt)    
    lcvx = pdef.RocketLanding(N, opt_params)
    problem = lcvx.problem4(min_d)
    
    import importlib
    module_name = f"lcvxP4_N{N}_cpg.cpg_solver"
    
    try:
        lcvxP4_solver = importlib.import_module(module_name)
        cpg_solve = lcvxP4_solver.cpg_solve
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"Solver module '{module_name}' not found. "
            f"Please generate the solver by running: python all_cpg.py {N}"
        )

    problem.register_solve('CPG', cpg_solve)
    problem.solve(method='CPG', verbose=False)
    
    return (tf_opt, lcvx, params.vessel_data)
    

def generate_problem3_solver(N):    
    from parameters import rocket_landing_parameters as p
    tf = 10
    params = p.Parameters(N)
    param_data = params.get_data(tf)
    lcvx = pdef.RocketLanding(N, param_data)
    problem = lcvx.problem3()

    cpg.generate_code(problem, code_dir=f'lcvxP3_N{N}_cpg', solver='CLARABEL', prefix='P3')


def generate_problem4_solver(N):
    from parameters import rocket_landing_parameters as p
    tf = 10
    min_d = 0
    params = p.Parameters(N)
    param_data = params.get_data(tf)
    
    lcvx = pdef.RocketLanding(N, param_data)
    problem = lcvx.problem4(min_d)

    cpg.generate_code(problem, code_dir=f'lcvxP4_N{N}_cpg', solver='CLARABEL', prefix='P4')