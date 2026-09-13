import numpy as np
import numpy.linalg as npl
from typing import Callable, Optional

# still WIP

def lerp(vec1, vec2, t):
    """Linear interpolation between two vectors."""
    return vec1 + t * (vec2 - vec1)


def clamp(num, min_val, max_val):
    """Restricts a number to be within a specified range."""
    return max(min(num, max_val), min_val)


def find_nearest_index(x_traj: np.ndarray, r_curr: np.ndarray, tf: float, N: int) -> float:
    """
    Finds the fractional index of the point on the trajectory closest to r_curr.
    Projects position error onto the velocity vector to get sub-index precision.
    Returns a floating point index in range approximately [0, N).
    """
    positions = x_traj[0:3, :]
    diff = positions - r_curr.reshape(3, 1)
    dists = npl.norm(diff, axis=0)
    nearest_i = int(np.argmin(dists))

    v_ref = x_traj[3:6, nearest_i]
    v_norm = npl.norm(v_ref)

    if v_norm < 1e-3:
        return float(nearest_i)

    v_dir = v_ref / v_norm
    vec_to_curr = r_curr - positions[:, nearest_i]
    dist_along = np.dot(vec_to_curr, v_dir)
    dt_node = tf / N
    frac = dist_along / (v_norm * dt_node)
    frac = clamp(frac, -0.5, 0.5)

    return nearest_i + frac


def sample_state(x_traj: np.ndarray, u_traj: np.ndarray, index: float, N: int):
    """Samples State (x) and Control (u) at a floating-point index.

    Returns (x_interp, u_interp) where each is a numpy array.
    """
    idx_int = int(np.floor(index))
    idx_frac = index - idx_int

    if idx_int < 0:
        return x_traj[:, 0], u_traj[:, 0]
    if idx_int >= N - 1:
        return x_traj[:, -1], u_traj[:, -1]

    x_interp = lerp(x_traj[:, idx_int], x_traj[:, idx_int + 1], idx_frac)
    u_interp = lerp(u_traj[:, idx_int], u_traj[:, idx_int + 1], idx_frac)

    return x_interp, u_interp


class GuidanceController:
    """Guidance controller extracted from the notebook.

    Usage:
        controller = GuidanceController(conn, vessel, landing_reference_frame,
                                        x_optimal, u_optimal, tf_val, v_data)
        controller.run()
    """

    def __init__(
        self,
        conn,
        vessel,
        landing_reference_frame,
        x_optimal: np.ndarray,
        u_optimal: np.ndarray,
        tf_val: float,
        v_data: dict,
        K_pos: float = 0.5,
        K_vel: float = 0.7,
        log_cb: Optional[Callable[[str], None]] = None,
    ):
        self.conn = conn
        self.vessel = vessel
        self.landing_reference_frame = landing_reference_frame
        self.x_optimal = x_optimal
        self.u_optimal = u_optimal
        self.tf_val = tf_val
        self.N_points = u_optimal.shape[1]
        self.max_thrust = v_data.get("thrust_max")
        self.g_vec = np.array(v_data.get("g", [0.0, 0.0, 0.0]))

        self.K_pos = K_pos
        self.K_vel = K_vel

        self.n_i = 0.0
        self.nav_mode = "gfold"

        self.pos_stream = None
        self.vel_stream = None
        self.log_cb = log_cb

    def _log(self, msg: str):
        if self.log_cb:
            try:
                self.log_cb(msg)
            except Exception:
                pass

    def create_streams(self):
        self._log("Initializing streams")
        self.pos_stream = self.conn.add_stream(self.vessel.position, self.landing_reference_frame)
        self.vel_stream = self.conn.add_stream(self.vessel.velocity, self.landing_reference_frame)

    def engage_autopilot(self):
        self.conn.krpc.paused = False
        self.vessel.auto_pilot.engage()
        self.vessel.auto_pilot.reference_frame = self.landing_reference_frame
        self.vessel.control.rcs = True
        self.vessel.control.throttle = 0

    def disengage_autopilot(self):
        try:
            self.vessel.control.throttle = 0
        except Exception:
            pass
        try:
            self.vessel.auto_pilot.disengage()
        except Exception:
            pass

    def run(self):
        """Run the guidance loop until KeyboardInterrupt."""
        self.create_streams()
        self.engage_autopilot()

        self._log(f"Controller Started. Mode: {self.nav_mode}")

        try:
            while True:
                r_curr = np.array(self.pos_stream())
                v_curr = np.array(self.vel_stream())
                mass = self.vessel.mass

                new_n_i = find_nearest_index(self.x_optimal, r_curr, self.tf_val, self.N_points)
                self.n_i = max(self.n_i, new_n_i)

                x_ref, u_ref = sample_state(self.x_optimal, self.u_optimal, self.n_i, self.N_points)
                r_ref = x_ref[0:3]
                v_ref = x_ref[3:6]

                err_r = r_ref - r_curr
                err_v = v_ref - v_curr

                accel_fb = (err_r * self.K_pos) + (err_v * self.K_vel)
                f_cmd = u_ref + (mass * accel_fb)

                f_mag = npl.norm(f_cmd)
                throttle_cmd = f_mag / self.max_thrust if self.max_thrust else 0.0
                throttle_cmd = clamp(throttle_cmd, 0.001, 1.0)

                if f_mag > 1.0:
                    pointing_cmd = tuple(f_cmd / f_mag)
                    self.vessel.auto_pilot.target_direction = pointing_cmd

                self.vessel.control.throttle = float(throttle_cmd)

        except KeyboardInterrupt:
            self._log("Manual Override.")
        finally:
            self.disengage_autopilot()
            self._log("Controller Disengaged.")
