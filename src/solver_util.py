import numpy as np
import trajectory_solver as tsolver
import plot
import json
import argparse
import os


class RocketTrajectorySolver:
    def __init__(self, conditions):
        self.gravity_constant = 9.80665
        self.mass_dry = conditions['mass_dry']
        self.mass_fuel = conditions['mass_fuel']
        self.mass_wet = self.mass_dry + self.mass_fuel
        self.thrust_lower_bound = conditions['thrust_max'] * conditions['throttle'][0]
        self.thrust_upper_bound = conditions['thrust_max'] * conditions['throttle'][1]
        self.Isp = conditions['Isp']
        self.alpha = 1 / (self.gravity_constant * self.Isp)
        self.initial_state = conditions['initial_state']  # consisting of 6 components (z,x,y) for position and velocity
        self.angle_gs_cot = 1 / np.tan(conditions['angle_gs'])
        self.angle_pt_cos = np.cos(conditions['angle_pt'])
        self.mass_dry_log = np.log(self.mass_dry)
        self.mass_wet_log = np.log(self.mass_wet)
        self.max_velocity = conditions['velocity_max']
        self.gravity_vector = conditions['g']  # also (z,x,y)
        self.landing_point = conditions['landing_point']

    def estimate_time(self, N, problem_type, rf):
        # golden section search to estimate flight time with lowest cost for trajectory calculations
        # since experimentally, the cost is generally a unimodal function of the time of flight
        # and in p3, the cost is the landing error, in p4, the cost is the usage of fuel
        golden_ratio = (np.sqrt(5) - 1) * 0.5
        thrust_lower = self.thrust_lower_bound
        thrust_upper = self.thrust_upper_bound
        initial_speed = np.linalg.norm(self.initial_state[3:6])
        max_fuel_burn_time = self.mass_fuel / (self.alpha * thrust_lower)
        min_dry_mass_time = self.mass_dry * np.linalg.norm(initial_speed) / thrust_upper
        time_lower_bound, time_upper_bound = min_dry_mass_time, max_fuel_burn_time
        
        while not (time_upper_bound - time_lower_bound) ** 2 <= 10:
            time_difference = (time_upper_bound - time_lower_bound) * golden_ratio
            time_1, time_2 = time_lower_bound + time_difference, time_upper_bound - time_difference
            cost_1, cost_2 = self.calculate_cost(time_1, N, problem_type, rf), \
                            self.calculate_cost(time_2, N, problem_type, rf)
            if cost_1 > cost_2:
                time_upper_bound = time_1
            elif cost_2 > cost_1:
                time_lower_bound = time_2

        optimal_time = (time_upper_bound + time_lower_bound) * 0.5
        return optimal_time

    def calculate_cost(self, time, N, problem_type, rf):
        origin = rf

        bundle_data = self.prepare_bundle_data(N, origin, time)
        obj_opt, x, u, m, s, z = tsolver.lcvx(N, problem_type, bundle_data)

        if not obj_opt:
            print(f'Cannot solve problem {problem_type}.')
            return 1e10

        if problem_type == 'p3':
            cost = np.linalg.norm(x[0:3, N - 1] - origin)
        elif problem_type == 'p4':
            cost = -z[0, N - 1]  # minimizing negative z(t), which is equivalent to maximizing the the final mass.

        return cost

    def prepare_bundle_data(self, N, final_position, flight_time):
        time_interval = flight_time / N
        alpha_dt = self.alpha * time_interval
        time_array = np.linspace(0, (N - 1) * time_interval, N)

        z0_term = self.mass_wet - self.alpha * self.thrust_upper_bound * time_array
        z0_term_inv = (1 / z0_term).reshape(1, N)
        z0_term_log = np.log(z0_term).reshape(1, N)

        initial_state = self.initial_state.reshape(6, 1)
        gravity = self.gravity_vector.reshape(3, 1)
        sparse_params = np.array((alpha_dt, self.max_velocity,
                                  self.angle_gs_cot, self.angle_pt_cos, 
                                  self.mass_wet_log, self.thrust_lower_bound, 
                                  self.thrust_upper_bound, flight_time))
        sparse_params = sparse_params.reshape(len(sparse_params), 1)
        
        return (initial_state, z0_term_inv, z0_term_log, gravity, final_position, sparse_params)

    def run(self, N):
        # solving p3 firstly for minimum landing error
        landing_location = self.landing_point
        flight_time = self.estimate_time(N, 'p3', landing_location)

        bundle_data = self.prepare_bundle_data(N, landing_location, flight_time)
        obj_opt, x, u, m, s, z = tsolver.lcvx(N, 'p3', bundle_data)

        if not obj_opt:
            print('Cannot solve problem p3.')
            return None
        
        # then solve p4 for minimum fuel consumption
        closest_destination = x[0:3, N - 1]
        flight_time = self.estimate_time(N, 'p4', closest_destination)

        bundle_data = self.prepare_bundle_data(N, closest_destination, flight_time)
        obj_opt, x, u, m, s, z = tsolver.lcvx(N, 'p4', bundle_data)

        if not obj_opt:
            print('Cannot solve problem p4.')
            return None

        print('Time Of Flight:', flight_time)
        return flight_time, x, u, m, s, z

def parse_arguments():
    parser = argparse.ArgumentParser(description='Rocket Trajectory Solver')
    parser.add_argument('-f', 
                        metavar='JSON',
                        type=str,
                        help='path to JSON file containing vessel parameters')
    parser.add_argument('-n', 
                        metavar='Positive Integer',
                        type=int,
                        default=20,
                        help='number of intervals (default: 20)')
    args = parser.parse_args()
    
    if not args.f:
        parser.print_help()
        exit(1)
    
    if args.n <= 0:
        parser.error("Number of intervals (N) must be a positive integer.")
    
    return args

if __name__ == '__main__':
    try:
        command_line_args = parse_arguments()
        N = command_line_args.n
        
        if not os.path.isfile(command_line_args.f):
            raise FileNotFoundError(f"JSON file not found. Please provide a valid file path.")
        
        with open(command_line_args.f, 'r') as file:
            vessel_data = json.load(file)
                
        vessel_data['landing_point'] = np.array(vessel_data['landing_point'])
        vessel_data['initial_state'] = np.array(vessel_data['initial_state'])
        vessel_data['g'] = np.array(vessel_data['g'])
        vessel_data['angle_gs'] = np.radians(vessel_data['angle_gs'])
        vessel_data['angle_pt'] = np.radians(vessel_data['angle_pt'])
        
        gfold = RocketTrajectorySolver(vessel_data)
        time_of_flight, x, u, m, s, z = gfold.run(N)
        plot.run(time_of_flight, x, u, m, s, z, vessel_data)
        
    except json.JSONDecodeError:
        raise json.JSONDecodeError("Invalid JSON format. Please ensure the file contains valid JSON data.")
    except ValueError:
        raise ValueError("Invalid value format in the JSON data.")
    except KeyError as e:
        raise Exception(f"Missing key '{e.args[0]}' in the JSON data.")
    except Exception as e:
        print("Error: ", e)
    