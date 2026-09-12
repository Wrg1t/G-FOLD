import cvxpy as cp
import numpy as np


class RocketLanding:
    def __init__(self, N, packed_data):
        self.N = N

        self.x0, self.z0_term_inv, self.z0_term_log, self.z1_term_log, self.g, self.rf, self.sparse_params = packed_data
        self.alpha_dt, self.V_max, self.y_gs_cot, self.p_cs_cos, self.m_wet_log, self.m_dry_log, self.r1, self.r2, self.dt = self.sparse_params

        self._initialize_paramterters()
        self._initialize_variables()
        self._define_constraints()

    def _initialize_paramterters(self):
        self.x0 = cp.Parameter(np.shape(self.x0), 'x0', self.x0)
        self.z0_term_inv = cp.Parameter(np.shape(self.z0_term_inv), 'z0_term_inv', self.z0_term_inv)
        self.z0_term_log = cp.Parameter(np.shape(self.z0_term_log), 'z0_term_log', self.z0_term_log)
        self.z1_term_log = cp.Parameter(np.shape(self.z1_term_log), 'z1_term_log', self.z1_term_log)
        self.g = cp.Parameter(np.shape(self.g), 'g', self.g)
        self.alpha_dt = cp.Parameter(np.shape(self.alpha_dt), 'alpha_dt', self.alpha_dt)
        self.V_max = cp.Parameter(np.shape(self.V_max), 'V_max', self.V_max)
        self.y_gs_cot = cp.Parameter(np.shape(self.y_gs_cot), 'y_gs_cot', self.y_gs_cot)
        self.p_cs_cos = cp.Parameter(np.shape(self.p_cs_cos), 'p_cs_cos', self.p_cs_cos)
        self.m_wet_log = cp.Parameter(np.shape(self.m_wet_log), 'm_wet_log', self.m_wet_log)
        self.m_dry_log = cp.Parameter(np.shape(self.m_dry_log), 'm_dry_log', self.m_dry_log)
        self.rf = cp.Parameter(np.shape(self.rf), 'rf', self.rf)
        self.r1 = cp.Parameter(np.shape(self.r1), 'r1', self.r1)
        self.r2 = cp.Parameter(np.shape(self.r2), 'r2', self.r2)
        self.dt = cp.Parameter(np.shape(self.dt), 'dt', self.dt)
        self.min_d = cp.Parameter(value=0, name='min_d')

    def _initialize_variables(self):
        # state vector (three position, three velocity)
        self.x = cp.Variable((6, self.N), name='var_x')
        # u = Tc/mass because Tc[:,n]/m[n] is not allowed by DCP
        self.u = cp.Variable((3, self.N), name='var_u')
        self.z = cp.Variable(self.N, name='var_z')  # z = ln(mass)
        self.s = cp.Variable(self.N, name='var_s')  # thrust slack

    def _define_constraints(self):
        self.con = []
        self.con += [self.x[:, 0] == self.x0[:, 0]]  # initial state (position and velocity)
        self.con += [self.x[3:6, self.N-1] == np.array([0, 0, 0])]  # safe and sound on the ground.
        self.con += [self.s[self.N-1] == 0]  # shut the engine at last
        self.con += [self.u[:, 0] == self.s[0] * np.array([1, 0, 0])]  # thrust direction starts upwards
        self.con += [self.u[:, self.N-1] == self.s[self.N-1] * np.array([1, 0, 0])]  # and ends upwards
        self.con += [self.z[0] == self.m_wet_log]  # convexified (7)
        self.con += [self.z[self.N-1] >= self.m_dry_log]
        self.con += [self.x[0, self.N-1] == self.rf[0]]

        # a workaround allowing it to be DPP-compliant
        self.var_g = cp.Variable(np.shape(self.g), 'var_g')
        self.con += [self.var_g == self.g]

        self.lambda_1 = cp.Variable(self.N, name='var_l1')
        self.lambda_2 = cp.Variable(self.N, name='var_l2')
        self.z0 = cp.Variable(self.N, name='z0')

        for n in range(0, self.N-1):
            self.con += [self.x[3:6, n+1] == self.x[3:6, n] + (self.dt * 0.5) * ((self.u[:, n] + self.var_g[:, 0]) + (self.u[:, n+1] + self.var_g[:, 0]))]
            self.con += [self.x[0:3, n+1] == self.x[0:3, n] + (self.dt * 0.5) * (self.x[3:6, n+1] + self.x[3:6, n])]  # leapfrog integration
            
            self.con += [cp.norm((self.x[0:3, n] - self.x[0:3, self.N-1])[1:3]) - self.y_gs_cot * (self.x[0, n] - self.x[0, self.N-1]) <= 0]  # glideslope cone
            self.con += [cp.norm(self.x[3:6, n]) <= self.V_max]  # velocity
            
            self.con += [self.z[n+1] == self.z[n] - (self.alpha_dt * 0.5) * (self.s[n] + self.s[n+1])]  # mass decreases
            self.con += [cp.norm(self.u[:, n]) <= self.s[n]]  # limit thrust
            self.con += [self.u[0, n] >= self.p_cs_cos * self.s[n]]  # thrust pointing constraint
            self.con += [self.z0_term_log[n] <= self.z[n], self.z[n] <= self.z1_term_log[n]]  # ensure the physical bounds on z are not violated
            
            self.con += [self.z0[n] == self.z0_term_log[n]]
            self.con += [self.lambda_1[n] == self.z0_term_inv[n] * (1 - (self.z[n] - self.z0[n]))]
            self.con += [self.lambda_2[n] == self.z0_term_inv[n] * (1 - (self.z[n] - self.z0[n]))]

            # taylor series as a great approximation to keep the convexity
            self.con += [self.s[n] >= self.r1 * self.lambda_1[n] + (self.z[n] - self.z0[n]) ** 2 * 0.5]  # thrust lower bound
            self.con += [self.s[n] <= self.r2 * self.lambda_2[n]]  # thrust upper bound

    def problem3(self):
        expression = cp.norm(self.x[0:3, self.N-1] - self.rf)  # minimize landing error
        objective = cp.Minimize(expression)
        problem = cp.Problem(objective, self.con)
        return problem

    def problem4(self, min_distance):
        self.min_d.value = min_distance
        self.con += [cp.norm(self.x[0:3, self.N-1] - self.rf) <= self.min_d]
        expression = self.z[self.N-1]  # minimize fuel consumption
        objective = cp.Maximize(expression)
        problem = cp.Problem(objective, self.con)
        return problem
