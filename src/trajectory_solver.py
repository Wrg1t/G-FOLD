import numpy as np
from cvxpy import *


def lcvx(N, pmark, packed_data):
    if pmark == 'p3':
        program = 3
    elif pmark == 'p4':
        program = 4

    x0, z0_term_inv, z0_term_log, g, rf, sparse_params = packed_data
    alpha_dt, V_max, y_gs_cot, p_cs_cos, m_wet_log, r1, r2, tf_ = sparse_params

    dt = tf_ / N  # Integration dt
    print('N = ', N)

    x = Variable((6, N), name='var_x')  # state vector (3position,3velocity)
    u = Variable((3, N), name='var_u')  # u = Tc/mass because Tc[:,n]/m[n] is not allowed by DCP
    z = Variable((1, N), name='var_z')  # z = ln(mass)
    s = Variable((1, N), name='var_s')  # thrust slack

    con = []  # constraints list
    con += [x[0:3:1, 0] == x0[0:3, 0]]  # initial position
    con += [x[3:6, 0] == x0[3:6, 0]]  # initial velocity
    con += [x[3:6, N-1] == np.array([0, 0, 0])]  # safe and sound on the ground.
    con += [s[0, N-1] == 0]  # shut the engine at last
    con += [u[:, 0] == s[0, 0] * np.array([1, 0, 0])]  # thrust direction starts upwards
    con += [u[:, N-1] == s[0, N-1] * np.array([1, 0, 0])]  # and ends upwards
    con += [z[0, 0] == m_wet_log]  # convexified (7)

    if program == 3:
        con += [x[0, N-1] == 0]
    elif program == 4:
        con += [x[0:3, N-1] == rf]

    for n in range(0, N-1):
        con += [x[3:6, n+1] == x[3:6, n] + (dt * 0.5) * ((u[:, n] + g[:, 0]) + (u[:, n+1] + g[:, 0]))]
        con += [x[0:3, n+1] == x[0:3, n] + (dt * 0.5) * (x[3:6, n+1] + x[3:6, n])]  # leapfrog integration
        con += [norm((x[0:3, n] - x[0:3, N-1])[1:3]) - y_gs_cot * (x[0, n] - x[0, N-1]) <= 0]  # glideslope cone
        con += [norm(x[3:6, n]) <= V_max]  # velocity
        con += [z[0, n+1] == z[0, n] - (alpha_dt * 0.5) * (s[0, n] + s[0, n+1])]  # mass decreases
        con += [norm(u[:, n]) <= s[0, n]]  # limit thrust
        con += [u[0, n] >= p_cs_cos * s[0, n]]  # thrust pointing constraint

        if n > 0:
            z0 = z0_term_log[0, n]
            mu_1 = r1 * (z0_term_inv[0, n])
            mu_2 = r2 * (z0_term_inv[0, n])
            # taylor series expansion as a great approximation to keep the convexity
            con += [s[0, n] >= mu_1 * (1 - (z[0, n] - z0) + (z[0, n] - z0) ** 2 * 0.5)]  # thrust lower bound
            con += [s[0, n] <= mu_2 * (1 - (z[0, n] - z0))]  # thrust upper bound

    if program == 3:
        print('-----------------------------')
        expression = norm(x[0:3, N-1] - rf)
        objective = Minimize(expression)
        problem = Problem(objective, con)
        print('solving p3')
        obj_opt = problem.solve(solver=ECOS, verbose=True, max_iters=1000000, warm_start=True)
        print('-----------------------------')

    elif program == 4:
        print('-----------------------------')
        expression = z[0, N-1]
        objective = Maximize(expression)
        problem = Problem(objective, con)
        print('solving p4')
        obj_opt = problem.solve(solver=SCS, verbose=True, max_iters=1000000, warm_start=True)
        print('-----------------------------')

    if program in {3, 4}:
        if z.value is not None:
            m = np.exp(z.value)
            return obj_opt, x.value, u.value, m, s.value, z.value
        else:
            return None, None, None, None, None, None
