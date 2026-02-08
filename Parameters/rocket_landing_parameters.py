import numpy as np
import json


class Parameters:
    def __init__(self, N):
        with open("Parameters/vessel_parameters_mars.json", 'r') as file:
            self.vessel_data = json.load(file)
            
        self.N = N

        self.vessel_data['landing_point'] = np.array(self.vessel_data['landing_point']).reshape(-1, 3)
        self.vessel_data['initial_state'] = np.array(self.vessel_data['initial_state']).reshape(-1, 6)
        self.vessel_data['g'] = np.array(self.vessel_data['g']).reshape(-1, 3)
        self.vessel_data['angle_gs'] = np.radians(self.vessel_data['angle_gs'])
        self.vessel_data['angle_pt'] = np.radians(self.vessel_data['angle_pt'])

        self.gravity_constant = 9.80665

        self.mass_dry = self.vessel_data['mass_dry']
        self.mass_fuel = self.vessel_data['mass_fuel']
        self.mass_wet = self.mass_dry + self.mass_fuel
        self.mass_dry_log = np.log(self.mass_dry)
        self.mass_wet_log = np.log(self.mass_wet)

        self.thrust_lower_bound = self.vessel_data['thrust_max'] * self.vessel_data['throttle'][0]
        self.thrust_upper_bound = self.vessel_data['thrust_max'] * self.vessel_data['throttle'][1]
        self.Isp = self.vessel_data['Isp']
        self.alpha = 1 / (self.gravity_constant * self.Isp)

        self.initial_state = self.vessel_data['initial_state']  # consisting of 6 components (z,x,y) for position and velocity
        self.landing_point = self.vessel_data['landing_point']

        self.angle_gs_cot = 1 / np.tan(self.vessel_data['angle_gs'])
        self.angle_pt_cos = np.cos(self.vessel_data['angle_pt'])

        self.max_velocity = self.vessel_data['velocity_max']
        self.gravity_vector = self.vessel_data['g']  # also (z,x,y)
        self.landing_point = self.vessel_data['landing_point']

        self.time_upper_bound = self.mass_fuel / (self.alpha * self.thrust_lower_bound)
        self.time_lower_bound = self.mass_dry * np.linalg.norm(self.initial_state[0][3:]) / self.thrust_upper_bound

    def get_data(self, tf):
        dt = tf / self.N
        
        final_position = self.landing_point
        alpha_dt = self.alpha * dt
        time_array = np.linspace(0, (self.N - 1) * dt, self.N)

        z0_term = self.mass_wet - self.alpha * self.thrust_upper_bound * time_array
            
        z0_term_inv = (1 / z0_term).reshape(1, self.N)
        z0_term_log = np.log(z0_term).reshape(1, self.N)
    
        initial_state = self.initial_state.reshape(6, 1)
        gravity = self.gravity_vector.reshape(3, 1)
        sparse_params = (alpha_dt, self.max_velocity,
                                  self.angle_gs_cot, self.angle_pt_cos,
                                  self.mass_wet_log, self.mass_dry_log, self.thrust_lower_bound,
                                  self.thrust_upper_bound, dt)
        
        return (initial_state, z0_term_inv, z0_term_log, gravity, final_position, sparse_params)
