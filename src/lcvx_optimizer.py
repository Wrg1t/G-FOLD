import numpy as np
import cvxpy as cp


class LcvxOptimizer:
    def __init__(self, N, problem_type, packed_data):
        # notation here follows the paper
        self.N = N
        self.problem_type = 3 if problem_type == 'p3' else 4
        self.packed_data = packed_data
        self._unpack_datas()
        self._initialize_variables()
        self._define_constraints()
        
    def _unpack_datas(self):
        self.x0, self.z0_term_inv, self.z0_term_log, self.g, self.rf, self.sparse_params = self.packed_data
        self.alpha_dt, self.V_max, self.y_gs_cot, self.p_cs_cos, self.m_wet_log, self.r1, self.r2, self.tf_ = self.sparse_params
        self.dt = self.tf_ / self.N

    def _initialize_variables(self):
        self.x = cp.Variable((6, self.N), name='var_x')  # state vector (3position, 3velocity)
        self.u = cp.Variable((3, self.N), name='var_u')  # u = Tc/mass because Tc[:,n]/m[n] is not allowed by DCP
        self.z = cp.Variable((1, self.N), name='var_z')  # z = ln(mass)
        self.s = cp.Variable((1, self.N), name='var_s')  # thrust slack

    def _define_constraints(self):
        self.con = []
        self.con += [self.x[0:3:1, 0] == self.x0[0:3, 0]]  # initial position
        self.con += [self.x[3:6, 0] == self.x0[3:6, 0]]  # initial velocity
        self.con += [self.x[3:6, self.N-1] == np.array([0, 0, 0])]  # safe and sound on the ground.
        self.con += [self.s[0, self.N-1] == 0]  # shut the engine at last
        self.con += [self.u[:, 0] == self.s[0, 0] * np.array([1, 0, 0])]  # thrust direction starts upwards
        self.con += [self.u[:, self.N-1] == self.s[0, self.N-1] * np.array([1, 0, 0])]  # and ends upwards
        self.con += [self.z[0, 0] == self.m_wet_log]  # convexified (7)

        if self.problem_type == 3:
            self.con += [self.x[0, self.N-1] == 0]
        elif self.problem_type == 4:
            self.con += [self.x[0:3, self.N-1] == self.rf]

        for n in range(0, self.N-1):
            self.con += [self.x[3:6, n+1] == self.x[3:6, n] + (self.dt * 0.5) * ((self.u[:, n] + self.g[:, 0]) + (self.u[:, n+1] + self.g[:, 0]))]
            self.con += [self.x[0:3, n+1] == self.x[0:3, n] + (self.dt * 0.5) * (self.x[3:6, n+1] + self.x[3:6, n])]  # leapfrog integration
            self.con += [cp.norm((self.x[0:3, n] - self.x[0:3, self.N-1])[1:3]) - self.y_gs_cot * (self.x[0, n] - self.x[0, self.N-1]) <= 0]  # glideslope cone
            self.con += [cp.norm(self.x[3:6, n]) <= self.V_max]  # velocity
            self.con += [self.z[0, n+1] == self.z[0, n] - (self.alpha_dt * 0.5) * (self.s[0, n] + self.s[0, n+1])]  # mass decreases
            self.con += [cp.norm(self.u[:, n]) <= self.s[0, n]]  # limit thrust
            self.con += [self.u[0, n] >= self.p_cs_cos * self.s[0, n]]  # thrust pointing constraint

            if n > 0:
                z0 = self.z0_term_log[0, n]
                mu_1 = self.r1 * (self.z0_term_inv[0, n])
                mu_2 = self.r2 * (self.z0_term_inv[0, n])
                # taylor expansion as a great approximation to keep the convexity
                self.con += [self.s[0, n] >= mu_1 * (1 - (self.z[0, n] - z0) + (self.z[0, n] - z0) ** 2 * 0.5)]  # thrust lower bound
                self.con += [self.s[0, n] <= mu_2 * (1 - (self.z[0, n] - z0))]  # thrust upper bound
                
    def solve(self):
        if self.problem_type == 3:
            expression = cp.norm(self.x[0:3, self.N-1] - self.rf)
            objective = cp.Minimize(expression)
            problem = cp.Problem(objective, self.con)
            self.obj_opt = problem.solve(solver='ECOS', verbose=True, max_iters=1000000, warm_start=True)

        elif self.problem_type == 4:
            expression = self.z[0, self.N-1]
            objective = cp.Maximize(expression)
            problem = cp.Problem(objective, self.con)
            self.obj_opt = problem.solve(solver='SCS', verbose=True, max_iters=1000000, warm_start=True)

        if self.problem_type in {3, 4}:
            if self.z.value is not None:
                m = np.exp(self.z.value)
                return self.obj_opt, self.x.value, self.u.value, m, self.s.value, self.z.value
            else:
                return None, None, None, None, None, None
        