import numpy as np
import contextlib
from scipy.spatial.transform import Rotation as R
from learned_ctrlr_opt.systems.robots import Robot
from dataclasses import dataclass
from typing import Dict, Union
from learned_ctrlr_opt.utils.dataset_utils import denormalize
try:
    from rotorpy.vehicles.crazyflie_params import quad_params as cf_quad_params
    # from rotorpy.vehicles.hummingbird_params import quad_params

    from rotorpy.environments import Environment
    from rotorpy.vehicles.multirotor import Multirotor

    # You will also need a controller (currently there is only one) that works for your vehicle.
    from rotorpy.controllers.quadrotor_control import SE3Control

    # And a trajectory generator
    from rotorpy.trajectories.hover_traj import HoverTraj
    from rotorpy.trajectories.circular_traj import CircularTraj
    from rotorpy.trajectories.lissajous_traj import TwoDLissajous

    # Also, worlds are how we construct obstacles. The following class contains methods related to constructing these maps.
    from rotorpy.world import World
except ImportError:
    print("RotorPY not installed, quad geom will not be available.")
    SE3Control = object
    TwoDLissajous = object
import sys

def eval_success(traj):
    if np.any(traj[-1][0:3] > 5) or np.any(traj[-1][0:3] < -5) or not np.any(traj[-1]):
        return False
    return True

@dataclass
class SE3ControlGains:
    kp_pos_x: float = 6.5  # np.array([6.5, 6.5, 15])
    kp_pos_y: float = 6.5  # np.array([6.5, 6.5, 15])
    kp_pos_z: float = 15  # np.array([6.5, 6.5, 15])
    kd_pos_x: float = 6.5  # np.array([4.0, 4.0, 9])
    kd_pos_y: float = 6.5
    kd_pos_z: float = 15
    kp_att: float = 544
    kd_att: float = 46.64

    def get_list(self):
        return np.array([self.kp_pos_x,
                self.kp_pos_y,
                self.kp_pos_z,
                self.kd_pos_x,
                self.kd_pos_y,
                self.kd_pos_z,
                self.kp_att,
                self.kd_att])

    @staticmethod
    def get_bounds():
        return np.array([[0.5, 20],
                         [0.5, 20],
                         [0.5, 30],
                         [0.5, 20],
                         [0.5, 20],
                         [0.5, 30],
                         [200, 600],
                         [30, 60]])

    @staticmethod
    def generate_random(gains_to_randomize):
        random_params = denormalize(np.random.rand(8), SE3ControlGains.get_bounds())
        params = SE3ControlGains().get_list()
        params[gains_to_randomize] = random_params[gains_to_randomize]
        return SE3ControlGains(*list(params))


# defaults are crazyflie parameters
# how wide should these bounds be?
@dataclass
class QuadrotorParams:
    mass: float = 0.03
    Ixx: float = 1.43e-5
    Iyy: float = 1.43e-5
    Izz: float = 2.89e-5
    k_eta: float = 2.3e-8  # thrust coefficient

    # Divide by 2 to ensure that at half max speed, quadrotor can fly
    @staticmethod
    def ensure_flies(mass, rotor_speed_max, k_eta):
        return k_eta * (rotor_speed_max/2)**2 > mass * 9.81/4

    def get_list(self):
        return np.array([self.mass, self.Ixx, self.Iyy, self.Izz, self.k_eta])

    @staticmethod
    def get_bounds():
        return np.array([[0.01, 1],
                         [1.00e-6, 1.00e-1],
                         [1.00e-6, 1.00e-1],
                         [1.00e-6, 1.00e-1],
                         [1e-8, 1e-5]])

    # Ensure thrust to weight ratio > 1, so it can fly
    @staticmethod
    def generate_random(gains_to_randomize):
        random_params = denormalize(np.random.rand(5), QuadrotorParams.get_bounds())
        params = QuadrotorParams().get_list()
        params[gains_to_randomize] = random_params[gains_to_randomize]
        while not QuadrotorParams.ensure_flies(params[0], 2500, params[4]):
            random_params = denormalize(np.random.rand(5), QuadrotorParams.get_bounds())
            params[gains_to_randomize] = random_params[gains_to_randomize]
        return QuadrotorParams(*list(params))

@dataclass
class CrazyFlieParams:
    mass: float = 0.03
    Ixx: float = 1.43e-5
    Iyy: float = 1.43e-5
    Izz: float = 2.89e-5
    k_eta: float = 2.3e-8  # thrust coefficient

    # Divide by 2 to ensure that at half max speed, quadrotor can fly
    @staticmethod
    def ensure_flies(mass, rotor_speed_max, k_eta):
        return k_eta * (rotor_speed_max/2)**2 > mass * 9.81/4

    def get_list(self):
        return np.array([self.mass, self.Ixx, self.Iyy, self.Izz, self.k_eta])

    @staticmethod
    def get_bounds():
        return np.array([[0.01, 0.1],
                         [1.00e-6, 1.00e-3],
                         [1.00e-6, 1.00e-3],
                         [1.00e-6, 1.00e-3],
                         [1e-8, 1e-6]])

    # Ensure thrust to weight ratio > 1, so it can fly
    @staticmethod
    def generate_random(gains_to_randomize):
        random_params = denormalize(np.random.rand(5), CrazyFlieParams.get_bounds())
        params = QuadrotorParams().get_list()
        params[gains_to_randomize] = random_params[gains_to_randomize]
        while not CrazyFlieParams.ensure_flies(params[0], 2500, params[4]):
            random_params = denormalize(np.random.rand(5), CrazyFlieParams.get_bounds())
            params[gains_to_randomize] = random_params[gains_to_randomize]
        return CrazyFlieParams(*list(params))

@dataclass
class CrazyFlieParamsTrain(CrazyFlieParams):
    @staticmethod
    def get_bounds():
        return np.array([[0.02, 0.09],
                   [2.00e-6, 9.00e-4],
                   [2.00e-6, 9.00e-4],
                   [2.00e-6, 9.00e-4],
                   [2e-8, 8e-7]])

    @staticmethod
    def generate_random(gains_to_randomize):
        random_params = denormalize(np.random.rand(5), CrazyFlieParamsTrain.get_bounds())
        params = QuadrotorParams().get_list()
        params[gains_to_randomize] = random_params[gains_to_randomize]
        while not CrazyFlieParamsTrain.ensure_flies(params[0], 2500, params[4]):
            random_params = denormalize(np.random.rand(5), CrazyFlieParamsTrain.get_bounds())
            params[gains_to_randomize] = random_params[gains_to_randomize]
        return CrazyFlieParamsTrain(*list(params))

# Assumes that the Train Params are a continuous box within the full params
@dataclass
class CrazyFlieParamsTest(CrazyFlieParams):
    @staticmethod
    def lower_half_bounds():
        return np.array([[CrazyFlieParams.get_bounds()[i, 0], CrazyFlieParamsTrain.get_bounds()[i, 0]] for i in range(5)])

    @staticmethod
    def upper_half_bounds():
        return np.array([[CrazyFlieParamsTrain.get_bounds()[i, 1], CrazyFlieParams.get_bounds()[i, 1]] for i in range(5)])

    @staticmethod
    def generate_random(gains_to_randomize, half=False):
        if half:
            lower = np.random.randint(2) * np.ones(len(gains_to_randomize))  # pick between 0, 1
        else:
            lower = np.random.randint(2, size=len(gains_to_randomize))
        bounds = []
        for k in range(5):
            if lower[k] == 0:
                bounds.append(CrazyFlieParamsTest.lower_half_bounds()[k])
            else:
                bounds.append(CrazyFlieParamsTest.upper_half_bounds()[k])
        bounds = np.array(bounds)
        random_params = denormalize(np.random.rand(5), bounds)
        params = QuadrotorParams().get_list()
        params[gains_to_randomize] = random_params[gains_to_randomize]
        count = 0
        while not CrazyFlieParamsTest.ensure_flies(params[0], 2500, params[4]) and count < 1000:
            random_params = denormalize(np.random.rand(5), bounds)
            params[gains_to_randomize] = random_params[gains_to_randomize]
            count += 1
        if count >= 1000:
            print(f"could not generate a flying drone with these bounds!")
        return CrazyFlieParamsTest(*list(params))


# A class allowing us to set control gains of the quad
class ParametrizedSE3Control(SE3Control):
    def __init__(self, quadrotor_params: Dict,
                 ctrl_params: Union[SE3ControlGains, np.ndarray]):
        super().__init__(quadrotor_params)
        if isinstance(ctrl_params, SE3ControlGains):
            ctrl_params = np.array(ctrl_params.get_list())
        self.ctrl_params = ctrl_params.flatten()

        self.kp_pos = np.array([ctrl_params[0], ctrl_params[1], ctrl_params[2]])
        self.kd_pos = np.array([ctrl_params[3], ctrl_params[4], ctrl_params[5]])
        self.kp_att = ctrl_params[6]
        self.kd_att = ctrl_params[7]




class QuadrotorSE3Control(Robot):
    ControllerParamsT = SE3ControlGains
    ParamsT = QuadrotorParams
    def __init__(self, params: QuadrotorParams,
                 gains_to_optimize,
                 trajectory_obj,
                 t_f):
        self.gains_to_optimize = gains_to_optimize
        self.trajectory_obj = trajectory_obj
        self.t_f = t_f
        # self.quad_params = cf_quad_params

        # TODO(hersh500): these bounds need to be much tighter.
        # Allows tracking error scaling to blow up and causes weird scaling.
        self.blank_world_def = {"bounds":{"extents":[-6, 6, -6, 6, -6, 6]}, "blocks":[]}
        self.params = params

    def sample_trajectory(self, dt):
        num_samples = int(self.t_f/dt)
        state_dim = 3 + 3 + 1 + 1
        sampled_traj = np.zeros((num_samples, state_dim))
        for t in np.arange(0, self.t_f, dt):
            traj_val = self.trajectory_obj.update(t)
            sampled_traj[int(t/dt)][0:3] = traj_val["x"]
            sampled_traj[int(t/dt)][3:6] = traj_val["x_dot"]
            sampled_traj[int(t/dt)][6] = traj_val["yaw"]
            sampled_traj[int(t/dt)][7] = traj_val["yaw_dot"]
        return sampled_traj

    def _create_params_dict(self):
        new_params_dict = dict(cf_quad_params)
        new_params_dict["mass"] = self.params.mass
        new_params_dict["Ixx"] = self.params.Ixx
        new_params_dict["Iyy"] = self.params.Iyy
        new_params_dict["Izz"] = self.params.Izz
        new_params_dict["k_eta"] = self.params.k_eta
        return new_params_dict

    # evaluate control gains on the given trajectory
    def evaluate_x(self, x, render_override=False, video_path=None):
        glob_random_state = np.random.get_state()
        np.random.seed(42)  # Make simulator deterministic
        try:
            params_dict = self._create_params_dict()
            ctrl_params_array = np.array(SE3ControlGains().get_list())
            ctrl_params_array[self.gains_to_optimize] = x
            sim_rate = 100
            controller = ParametrizedSE3Control(cf_quad_params, ctrl_params_array)
            world = World(self.blank_world_def)
            sim_instance = Environment(vehicle=Multirotor(params_dict),
                                       controller=controller,
                                       trajectory=self.trajectory_obj,
                                       sim_rate=sim_rate,
                                       imu=None,
                                       mocap=None,
                                       estimator=None,
                                       world=world,
                                       safety_margin=0.25
                                       )

            x0 = {'x': np.array([0,0,0]),
                  'v': np.zeros(3,),
                  'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
                  'w': np.zeros(3,),
                  'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                  'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}

            sim_instance.vehicle.initial_state = x0
            results = sim_instance.run(t_final=self.t_f,  # The maximum duration of the environment in seconds
                                       use_mocap=False,
                                       terminate=None,
                                       plot=render_override,  # Boolean: plots the vehicle states and commands
                                       plot_mocap=False,  # Boolean: plots the motion capture pose and twist measurements
                                       plot_estimator=False,
                                       # Boolean: plots the estimator filter states and covariance diagonal elements
                                       plot_imu=False,  # Boolean: plots the IMU measurements
                                       animate_bool=render_override,
                                       # Boolean: determines if the animation of vehicle state will play.
                                       animate_wind=False,
                                       # Boolean: determines if the animation will include a scaled wind vector to indicate the local wind acting on the UAV.
                                       verbose=render_override,  # Boolean: will print statistics regarding the simulation.
                                       fname=video_path)
            x = results["state"]["x"]
            v = results["state"]["v"]
            w = results["state"]["w"]
            eulers = R.from_quat(results["state"]["q"]).as_euler("xyz")

            x_flat = results["flat"]["x"]
            yaw_flat = results["flat"]["yaw"]
            rotor_speeds = results["state"]["rotor_speeds"]
            cmd_thrust = results["control"]["cmd_thrust"]

            # Does this aggregate too many terms together?
            pos_error = np.sum(np.linalg.norm(x - x_flat, axis=1))/x.shape[0]
            yaw_error = np.sum(np.linalg.norm(eulers[:,2]- yaw_flat, axis=0))/x.shape[0]
            pitchroll = np.sum(np.linalg.norm(eulers[:,:2], axis=1))/x.shape[0]
            # effort = np.sum(rotor_speeds)/x.shape[0]
            effort = np.sum(np.abs(cmd_thrust))/x.shape[0]
            return np.array([pos_error, yaw_error, pitchroll, effort])
        finally:
            np.random.set_state(glob_random_state)

    def evaluate_x_return_traj_stochastic(self, x, render_override=False, video_path=None):
        params_dict = self._create_params_dict()
        ctrl_params_array = np.array(SE3ControlGains().get_list())
        ctrl_params_array[self.gains_to_optimize] = x
        sim_rate = 100
        controller = ParametrizedSE3Control(cf_quad_params, ctrl_params_array)
        world = World(self.blank_world_def)
        sim_instance = Environment(vehicle=Multirotor(params_dict),
                                   controller=controller,
                                   trajectory=self.trajectory_obj,
                                   sim_rate=sim_rate,
                                   imu=None,
                                   mocap=None,
                                   estimator=None,
                                   world=world,
                                   safety_margin=0.25
                                   )

        x0 = {'x': np.array([0,0,0]),
              'v': np.zeros(3,),
              'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
              'w': np.zeros(3,),
              'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
              'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}

        sim_instance.vehicle.initial_state = x0
        results = sim_instance.run(t_final=self.t_f,  # The maximum duration of the environment in seconds
                                   use_mocap=False,
                                   terminate=None,
                                   plot=render_override,  # Boolean: plots the vehicle states and commands
                                   plot_mocap=False,  # Boolean: plots the motion capture pose and twist measurements
                                   plot_estimator=False,
                                   # Boolean: plots the estimator filter states and covariance diagonal elements
                                   plot_imu=False,  # Boolean: plots the IMU measurements
                                   animate_bool=render_override,
                                   # Boolean: determines if the animation of vehicle state will play.
                                   animate_wind=False,
                                   # Boolean: determines if the animation will include a scaled wind vector to indicate the local wind acting on the UAV.
                                   verbose=render_override,  # Boolean: will print statistics regarding the simulation.
                                   fname=video_path)
        x = results["state"]["x"]
        v = results["state"]["v"]
        w = results["state"]["w"]
        eulers = R.from_quat(results["state"]["q"]).as_euler("xyz")
        actual_traj = np.zeros((self.t_f*sim_rate, 3+3+2))  # save x, y, z, velocity, yaw+velocity
        lim = min(x.shape[0], self.t_f*sim_rate)
        actual_traj[:lim,0:3] = x[:lim]
        actual_traj[:lim,3:6] = v[:lim]
        actual_traj[:lim,6] = eulers[:lim,2]
        actual_traj[:lim,7] = results["state"]["w"][:lim,2]

        x_flat = results["flat"]["x"]
        yaw_flat = results["flat"]["yaw"]
        rotor_speeds = results["state"]["rotor_speeds"]
        cmd_thrust = results["control"]["cmd_thrust"]

        # Does this aggregate too many terms together?
        pos_error = np.sum(np.linalg.norm(x - x_flat, axis=1))/x.shape[0]
        yaw_error = np.sum(np.linalg.norm(eulers[:,2]- yaw_flat, axis=0))/x.shape[0]
        pitchroll = np.sum(np.linalg.norm(eulers[:,:2], axis=1))/x.shape[0]
        effort = np.sum(np.abs(cmd_thrust))/x.shape[0]
        return np.array([pos_error, yaw_error, pitchroll, effort]), actual_traj

    def evaluate_x_return_traj(self, x, render_override=False, video_path=None):
        glob_random_state = np.random.get_state()
        np.random.seed(42)  # Make simulator deterministic
        try:
            params_dict = self._create_params_dict()
            ctrl_params_array = np.array(SE3ControlGains().get_list())
            ctrl_params_array[self.gains_to_optimize] = x
            sim_rate = 100
            controller = ParametrizedSE3Control(cf_quad_params, ctrl_params_array)
            world = World(self.blank_world_def)
            sim_instance = Environment(vehicle=Multirotor(params_dict),
                                       controller=controller,
                                       trajectory=self.trajectory_obj,
                                       sim_rate=sim_rate,
                                       imu=None,
                                       mocap=None,
                                       estimator=None,
                                       world=world,
                                       safety_margin=0.25
                                       )

            x0 = {'x': np.array([0,0,0]),
                  'v': np.zeros(3,),
                  'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
                  'w': np.zeros(3,),
                  'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                  'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}

            sim_instance.vehicle.initial_state = x0
            results = sim_instance.run(t_final=self.t_f,  # The maximum duration of the environment in seconds
                                       use_mocap=False,
                                       terminate=None,
                                       plot=render_override,  # Boolean: plots the vehicle states and commands
                                       plot_mocap=False,  # Boolean: plots the motion capture pose and twist measurements
                                       plot_estimator=False,
                                       # Boolean: plots the estimator filter states and covariance diagonal elements
                                       plot_imu=False,  # Boolean: plots the IMU measurements
                                       animate_bool=render_override,
                                       # Boolean: determines if the animation of vehicle state will play.
                                       animate_wind=False,
                                       # Boolean: determines if the animation will include a scaled wind vector to indicate the local wind acting on the UAV.
                                       verbose=render_override,  # Boolean: will print statistics regarding the simulation.
                                       fname=video_path)
            x = results["state"]["x"]
            v = results["state"]["v"]
            w = results["state"]["w"]
            eulers = R.from_quat(results["state"]["q"]).as_euler("xyz")
            actual_traj = np.zeros((self.t_f*sim_rate, 3+3+2))  # save x, y, z, velocity, yaw+velocity
            lim = min(x.shape[0], self.t_f*sim_rate)
            actual_traj[:lim,0:3] = x[:lim]
            actual_traj[:lim,3:6] = v[:lim]
            actual_traj[:lim,6] = eulers[:lim,2]
            actual_traj[:lim,7] = results["state"]["w"][:lim,2]

            x_flat = results["flat"]["x"]
            yaw_flat = results["flat"]["yaw"]
            rotor_speeds = results["state"]["rotor_speeds"]
            cmd_thrust = results["control"]["cmd_thrust"]

            # Does this aggregate too many terms together?
            pos_error = np.sum(np.linalg.norm(x - x_flat, axis=1))/x.shape[0]
            yaw_error = np.sum(np.linalg.norm(eulers[:,2]- yaw_flat, axis=0))/x.shape[0]
            pitchroll = np.sum(np.linalg.norm(eulers[:,:2], axis=1))/x.shape[0]
            effort = np.sum(np.abs(cmd_thrust))/x.shape[0]
            return np.array([pos_error, yaw_error, pitchroll, effort]), actual_traj
        finally:
            np.random.set_state(glob_random_state)


    def get_gain_bounds(self):
        return np.array([])

    def get_nominal_values(self):
        return np.array(SE3ControlGains().get_list())[self.gains_to_optimize]

    def get_theta_bounds(self):
        return np.array([])

    @staticmethod
    def perf_metric_names():
        return ["Linear Tracking Error", "Heading Error", "Pitch/Roll", "Effort"]

    @staticmethod
    def random_circ_traj(seed, radius_bounds, freq_bounds):
        np.random.seed(seed)
        rad = np.random.rand() * (radius_bounds[1] - radius_bounds[0]) + radius_bounds[0]
        freq = np.random.rand() * (freq_bounds[1] - freq_bounds[0]) + freq_bounds[0]
        # yaw = True if np.random.rand() > 0.5 else False
        yaw = True # yaw is currently broken in simulator?
        plane = "XY" if np.random.rand() > 0.5 else "XZ"
        direction = "CCW" if np.random.rand() > 0.5 else "CW"

        traj_obj = CircularTraj(radius=rad, freq=freq, yaw_bool=yaw, direction=direction)
        dir_int = 1 if direction == "CCW" else 0
        plane_int = 1 if plane == "XY" else 0
        # return traj_obj, [rad, freq, int(yaw), dir_int]
        return traj_obj, [rad, freq, plane_int, dir_int]

    @staticmethod
    def random_xy_ellipse_traj(seed, radius_bounds, freq_bounds):
        np.random.seed(seed)
        rad = np.random.rand(2) * (radius_bounds[1] - radius_bounds[0]) + radius_bounds[0]
        freq = np.random.rand() * (freq_bounds[1] - freq_bounds[0]) + freq_bounds[0]
        center = np.array([0, 0, 0])
        radius = np.array([rad[0], rad[1], 0])
        frequency = np.array([freq, freq, 0])
        yaw = False
        traj_obj = ThreeDCircularTraj_fixed(radius=radius, freq=frequency, yaw_bool=yaw)
        return traj_obj, [rad[0], rad[1], freq]

    def get_thetas(self):
        return self.params.get_list()


class QuadrotorSE3Control_ResetFree(QuadrotorSE3Control):
    def evaluate_x(self, x, initial_state=None, render=False, wind=None):
        params_dict = self._create_params_dict()
        ctrl_params_array = np.array(SE3ControlGains().get_list())
        ctrl_params_array[self.gains_to_optimize] = x
        sim_rate = 100
        world = World(self.blank_world_def)
        controller = ParametrizedSE3Control(cf_quad_params, ctrl_params_array)
        sim_instance = Environment(vehicle=Multirotor(params_dict),
                                   controller=controller,
                                   trajectory=self.trajectory_obj,
                                   sim_rate=sim_rate,
                                   imu=None,
                                   mocap=None,
                                   estimator=None,
                                   world=world,
                                   safety_margin=0.25,
                                   wind_profile=wind
                                   )
        if initial_state is None:
            x0 = {'x': np.array([0,0,0]),
                  'v': np.zeros(3,),
                  'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
                  'w': np.zeros(3,),
                  'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                  'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}
        else:
            x0 = {'x': initial_state["state"]["x"][-1],
                  'v': initial_state["state"]["v"][-1],
                  'q': initial_state["state"]["q"][-1],
                  'w': initial_state["state"]["w"][-1],
                  'wind': np.array([0, 0, 0]),
                  'rotor_speeds': initial_state["state"]["rotor_speeds"][-1]}

        sim_instance.vehicle.initial_state = x0
        results = sim_instance.run(t_final=self.t_f,  # The maximum duration of the environment in seconds
                                   use_mocap=False,
                                   terminate=None,
                                   plot=False,  # Boolean: plots the vehicle states and commands
                                   plot_mocap=False,  # Boolean: plots the motion capture pose and twist measurements
                                   plot_estimator=False,
                                   plot_imu=False,  # Boolean: plots the IMU measurements
                                   animate_bool=render,
                                   animate_wind=False,
                                   verbose=render)

        x = results["state"]["x"]
        v = results["state"]["v"]
        w = results["state"]["w"]
        eulers = R.from_quat(results["state"]["q"]).as_euler("xyz")

        x_flat = results["flat"]["x"]
        v_flat = results["flat"]["x_dot"]
        # print("----X_FLAT----")
        # print(x_flat)
        # print("----X----")
        # print(x)
        yaw_flat = results["flat"]["yaw"]
        rotor_speeds = results["state"]["rotor_speeds"]
        cmd_thrust = results["control"]["cmd_thrust"]

        actual_traj = np.zeros((self.t_f*sim_rate, 3+3+2+2+1))
        lim = min(x.shape[0], self.t_f*sim_rate)  # in case of premature failure
        actual_traj[:lim,0:3] = x[:lim] - x_flat[:lim]  # output tracking error instead of just raw positions
        actual_traj[:lim,3:6] = v[:lim] - v_flat[:lim]
        actual_traj[:lim,6] = eulers[:lim,2]
        actual_traj[:lim,7:10] = w[:lim]
        # actual_traj[:lim,8:] = rotor_speeds[:lim]
        actual_traj[:lim,10] = cmd_thrust[:lim]

        # Does this aggregate too many terms together?
        pos_error = np.sum(np.linalg.norm(x - x_flat, axis=1))/x.shape[0]
        yaw_error = np.sum(np.linalg.norm(eulers[:,2]- yaw_flat, axis=0))/x.shape[0]
        # pitchroll = np.sum(np.linalg.norm(eulers[:,:2], axis=1))/x.shape[0]
        pitchroll = np.sum(np.linalg.norm(w[:2], axis=1))/x.shape[0]   # output pitchroll velocity instead...
        effort = np.sum(np.abs(cmd_thrust))/x.shape[0]
        self.trajectory_obj.shift_forward(self.t_f)
        return np.array([pos_error, yaw_error, pitchroll, effort]), actual_traj, results


class QuadrotorSE3Control_RandomStart(QuadrotorSE3Control):
    def __init__(self, params: QuadrotorParams,
                 gains_to_optimize,
                 trajectory_obj,
                 t_f,
                 initial_gain,
                 initial_tf,
                 initial_sensor_traj_length):
        super().__init__(params, gains_to_optimize, trajectory_obj, t_f)
        self.initial_gain = initial_gain
        self.initial_tf = initial_tf
        self.initial_sensor_traj_length = initial_sensor_traj_length

    # evaluate control gains on the given trajectory
    def evaluate_x(self, x, render_override=False, video_path=None):
        params_dict = self._create_params_dict()
        initial_ctrl_params_array = np.array(SE3ControlGains().get_list())
        initial_ctrl_params_array[self.gains_to_optimize] = self.initial_gain
        ctrl_params_array = np.array(SE3ControlGains().get_list())
        ctrl_params_array[self.gains_to_optimize] = x
        sim_rate = 100

        initial_controller = ParametrizedSE3Control(cf_quad_params, initial_ctrl_params_array)
        controller = ParametrizedSE3Control(cf_quad_params, ctrl_params_array)
        world = World(self.blank_world_def)
        sim_instance = Environment(vehicle=Multirotor(params_dict),
                                   controller=initial_controller,
                                   trajectory=self.trajectory_obj,
                                   sim_rate=sim_rate,
                                   imu=None,
                                   mocap=None,
                                   estimator=None,
                                   world=world,
                                   safety_margin=0.25
                                   )

        x0 = {'x': np.array([0,0,0]),
              'v': np.zeros(3,),
              'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
              'w': np.zeros(3,),
              'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
              'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}

        sim_instance.vehicle.initial_state = x0
        initial_results = sim_instance.run(t_final=self.initial_tf,  # The maximum duration of the environment in seconds
                                   use_mocap=False,
                                   terminate=None,
                                   plot=False,  # Boolean: plots the vehicle states and commands
                                   plot_mocap=False,  # Boolean: plots the motion capture pose and twist measurements
                                   plot_estimator=False,
                                   plot_imu=False,  # Boolean: plots the IMU measurements
                                   animate_bool=render_override,
                                   animate_wind=False,
                                   verbose=render_override)
        initial_x = initial_results["state"]["x"]
        initial_v = initial_results["state"]["v"]
        initial_w = initial_results["state"]["w"]
        initial_eulers = R.from_quat(initial_results["state"]["q"]).as_euler("xyz")

        initial_x_flat = initial_results["flat"]["x"]
        initial_v_flat = initial_results["flat"]["x_dot"]
        initial_yaw_flat = initial_results["flat"]["yaw"]
        initial_rotor_speeds = initial_results["state"]["rotor_speeds"]
        initial_thrusts = initial_results["control"]["cmd_thrust"]

        # This forms the initial trajectory
        # First do Oob or safety checks on the initial trajectory
        if np.any(initial_x[-1] > 10) or np.any(initial_x[-1] < -10):
            return False, None, None, None
        if len(initial_x) < self.initial_sensor_traj_length:
            print(f"Initial x is shorter than the specified sensor traj length: {len(initial_x)}")
            print(initial_x)

        # Next, do another sim instance and run the actual controller
        starting_state = {'x': initial_x[-1],
                          'v': initial_v[-1],
                          'q': initial_results["state"]["q"][-1],
                          'w': initial_w[-1],
                          'wind': np.array([0, 0, 0]),
                          'rotor_speeds': initial_rotor_speeds[-1]}

        self.trajectory_obj.shift_forward(self.initial_tf)
        sim_instance = Environment(vehicle=Multirotor(params_dict),
                                   controller=controller,
                                   trajectory=self.trajectory_obj,
                                   sim_rate=sim_rate,
                                   imu=None,
                                   mocap=None,
                                   estimator=None,
                                   world=world,
                                   safety_margin=0.25
                                   )
        sim_instance.vehicle.initial_state = starting_state

        results = sim_instance.run(t_final=self.t_f,  # The maximum duration of the environment in seconds
                                   use_mocap=False,
                                   terminate=None,
                                   plot=render_override,  # Boolean: plots the vehicle states and commands
                                   plot_mocap=False,  # Boolean: plots the motion capture pose and twist measurements
                                   plot_estimator=False,
        #                            Boolean: plots the estimator filter states and covariance diagonal elements
                                   plot_imu=False,  # Boolean: plots the IMU measurements
                                   animate_bool=render_override,
                                   # Boolean: determines if the animation of vehicle state will play.
                                   animate_wind=False,
                                   # Boolean: determines if the animation will include a scaled wind vector to indicate the local wind acting on the UAV.
                                   verbose=render_override,  # Boolean: will print statistics regarding the simulation.
                                   fname=video_path)

        x = results["state"]["x"]
        v = results["state"]["v"]
        w = results["state"]["w"]
        eulers = R.from_quat(results["state"]["q"]).as_euler("xyz")

        x_flat = results["flat"]["x"]
        v_flat = results["flat"]["x_dot"]
        yaw_flat = results["flat"]["yaw"]
        rotor_speeds = results["state"]["rotor_speeds"]
        cmd_thrust = results["control"]["cmd_thrust"]

        try:
            initial_traj = np.zeros((self.initial_sensor_traj_length, 3+3+2+2+1))
            initial_traj[:,0:3] = initial_x[-self.initial_sensor_traj_length:] - initial_x_flat[-self.initial_sensor_traj_length:]
            initial_traj[:,3:6] = initial_v[-self.initial_sensor_traj_length:] - initial_v_flat[-self.initial_sensor_traj_length:]
            initial_traj[:,6] = initial_eulers[-self.initial_sensor_traj_length:,2]
            initial_traj[:,7:10] = initial_w[-self.initial_sensor_traj_length:]
            initial_traj[:,10] = initial_thrusts[-self.initial_sensor_traj_length:] # initial_rotor_speeds[-self.initial_sensor_traj_length:]
        except ValueError:
            return False, None, None, None

        actual_traj = np.zeros((self.t_f*sim_rate, 3+3+2+2+1))
        lim = min(x.shape[0], self.t_f*sim_rate)
        actual_traj[:lim,0:3] = x[:lim] - x_flat[:lim]
        actual_traj[:lim,3:6] = v[:lim] - v_flat[:lim]
        actual_traj[:lim,6] = eulers[:lim,2]
        actual_traj[:lim,7:10] = w[:lim]
        actual_traj[:lim,10] = cmd_thrust[:lim] # rotor_speeds[:lim]

        # Does this aggregate too many terms together?
        pos_error = np.sum(np.linalg.norm(x - x_flat, axis=1))/x.shape[0]
        yaw_error = np.sum(np.linalg.norm(eulers[:,2]- yaw_flat, axis=0))/x.shape[0]
        # pitchroll = np.sum(np.linalg.norm(eulers[:,:2], axis=1))/x.shape[0]
        pitchroll = np.sum(np.linalg.norm(w[:2], axis=1))/x.shape[0]   # output pitchroll velocity instead...
        effort = np.sum(np.abs(cmd_thrust))/x.shape[0]
        return True, np.array([pos_error, yaw_error, pitchroll, effort]), actual_traj, initial_traj

    def evaluate_x_initial(self):
        raise DeprecationWarning("Not updated to give tracking error in trajectory")
        params_dict = self._create_params_dict()
        initial_ctrl_params_array = np.array(SE3ControlGains().get_list())
        initial_ctrl_params_array[self.gains_to_optimize] = self.initial_gain
        sim_rate = 100
        # TODO(hersh500): This initial controller knows the true parameters of the quad...
        initial_controller = ParametrizedSE3Control(cf_quad_params, initial_ctrl_params_array)
        world = World(self.blank_world_def)
        sim_instance = Environment(vehicle=Multirotor(params_dict),
                                   controller=initial_controller,
                                   trajectory=self.trajectory_obj,
                                   sim_rate=sim_rate,
                                   imu=None,
                                   mocap=None,
                                   estimator=None,
                                   world=world,
                                   safety_margin=0.25
                                   )

        x0 = {'x': np.array([0,0,0]),
              'v': np.zeros(3,),
              'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
              'w': np.zeros(3,),
              'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
              'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}

        sim_instance.vehicle.initial_state = x0
        initial_results = sim_instance.run(t_final=self.initial_tf,  # The maximum duration of the environment in seconds
                                           use_mocap=False,
                                           terminate=None,
                                           plot=False,  # Boolean: plots the vehicle states and commands
                                           plot_mocap=False,  # Boolean: plots the motion capture pose and twist measurements
                                           plot_estimator=False,
                                           plot_imu=False,  # Boolean: plots the IMU measurements
                                           animate_bool=False,
                                           animate_wind=False,
                                           verbose=False)

        initial_x = initial_results["state"]["x"]
        initial_v = initial_results["state"]["v"]
        initial_w = initial_results["state"]["w"]
        initial_eulers = R.from_quat(initial_results["state"]["q"]).as_euler("xyz")

        initial_x_flat = initial_results["flat"]["x"]
        initial_yaw_flat = initial_results["flat"]["yaw"]
        initial_rotor_speeds = initial_results["state"]["rotor_speeds"]
        initial_thrusts = initial_results["control"]["cmd_thrust"]

        # This forms the initial trajectory
        # First do Oob or safety checks on the initial trajectory
        if np.any(initial_x[-1] > 10) or np.any(initial_x[-1] < -10):
            return False, None, None

        try:
            initial_traj = np.zeros((self.initial_sensor_traj_length, 3+3+2+1))
            initial_traj[:,0:3] = initial_x[-self.initial_sensor_traj_length:]
            initial_traj[:,3:6] = initial_v[-self.initial_sensor_traj_length:]
            initial_traj[:,6] = initial_eulers[-self.initial_sensor_traj_length:,2]
            initial_traj[:,7] = initial_w[-self.initial_sensor_traj_length:,2]
            initial_traj[:,8] = initial_thrusts[-self.initial_sensor_traj_length:]  # initial_rotor_speeds[-self.initial_sensor_traj_length:]
        except ValueError:
            return False, None, None
        return True, initial_traj, initial_results

    def evaluate_x_post(self, x, initial_results):
        params_dict = self._create_params_dict()
        ctrl_params_array = np.array(SE3ControlGains().get_list())
        ctrl_params_array[self.gains_to_optimize] = x
        sim_rate = 100

        # TODO(hersh500): This initial controller knows the true parameters of the quad...
        # Is this actually used anywhere?
        controller = ParametrizedSE3Control(cf_quad_params, ctrl_params_array)
        world = World(self.blank_world_def)
        initial_x = initial_results["state"]["x"]
        initial_v = initial_results["state"]["v"]
        initial_w = initial_results["state"]["w"]
        initial_eulers = R.from_quat(initial_results["state"]["q"]).as_euler("xyz")

        initial_x_flat = initial_results["flat"]["x"]
        initial_yaw_flat = initial_results["flat"]["yaw"]
        initial_rotor_speeds = initial_results["state"]["rotor_speeds"]

        starting_state = {'x': initial_x[-1],
                          'v': initial_v[-1],
                          'q': initial_results["state"]["q"][-1],
                          'w': initial_w[-1],
                          'wind': np.array([0, 0, 0]),
                          'rotor_speeds': initial_rotor_speeds[-1]}

        self.trajectory_obj.shift_forward(self.initial_tf)
        sim_instance = Environment(vehicle=Multirotor(params_dict),
                                   controller=controller,
                                   trajectory=self.trajectory_obj,
                                   sim_rate=sim_rate,
                                   imu=None,
                                   mocap=None,
                                   estimator=None,
                                   world=world,
                                   safety_margin=0.25
                                   )
        sim_instance.vehicle.initial_state = starting_state

        results = sim_instance.run(t_final=self.t_f,  # The maximum duration of the environment in seconds
                                   use_mocap=False,
                                   terminate=None,
                                   plot=False,  # Boolean: plots the vehicle states and commands
                                   plot_mocap=False,  # Boolean: plots the motion capture pose and twist measurements
                                   plot_estimator=False,
                                   #                            Boolean: plots the estimator filter states and covariance diagonal elements
                                   plot_imu=False,  # Boolean: plots the IMU measurements
                                   animate_bool=False,
                                   # Boolean: determines if the animation of vehicle state will play.
                                   animate_wind=False,
                                   # Boolean: determines if the animation will include a scaled wind vector to indicate the local wind acting on the UAV.
                                   verbose=False)  # Boolean: will print statistics regarding the simulation.

        x = results["state"]["x"]
        v = results["state"]["v"]
        w = results["state"]["w"]
        eulers = R.from_quat(results["state"]["q"]).as_euler("xyz")

        x_flat = results["flat"]["x"]
        yaw_flat = results["flat"]["yaw"]
        rotor_speeds = results["state"]["rotor_speeds"]
        cmd_thrust = initial_results["control"]["cmd_thrust"]

        actual_traj = np.zeros((self.t_f*sim_rate, 3+3+2+1))
        lim = min(x.shape[0], self.t_f*sim_rate)
        actual_traj[:lim,0:3] = x[:lim]
        actual_traj[:lim,3:6] = v[:lim]
        actual_traj[:lim,6] = eulers[:lim,2]
        actual_traj[:lim,7] = w[:lim,2]
        actual_traj[:lim,8] = cmd_thrust[:lim]  # rotor_speeds[:lim]

        # Does this aggregate too many terms together?
        pos_error = np.sum(np.linalg.norm(x - x_flat, axis=1))/x.shape[0]
        yaw_error = np.sum(np.linalg.norm(eulers[:,2]- yaw_flat, axis=0))/x.shape[0]
        pitchroll = np.sum(np.linalg.norm(eulers[:,:2], axis=1))/x.shape[0]
        effort = np.sum(np.abs(cmd_thrust))/x.shape[0]
        return np.array([pos_error, yaw_error, pitchroll, effort]), actual_traj


    def evaluate_x_return_traj_stochastic(self, x, render_override=False, video_path=None):
        raise NotImplementedError("Use evaluate_x() instead.")

    def evaluate_x_return_traj(self, x, render_override=False, video_path=None):
        raise NotImplementedError("Use evaluate_x() instead.")


class CrazyFlieSE3Control(QuadrotorSE3Control):
    ControllerParamsT = SE3ControlGains
    ParamsT = CrazyFlieParams


class CrazyFlieSE3Control_RandomStart_Train(QuadrotorSE3Control_RandomStart):
    ControllerParamsT = SE3ControlGains
    ParamsT = CrazyFlieParamsTrain


class CrazyFlieSE3Control_ResetFree(QuadrotorSE3Control_ResetFree):
    ControllerParamsT = SE3ControlGains
    ParamsT = CrazyFlieParams


def generate_random_circle_traj(freq_limit):
    frequencies = np.random.rand(3) * freq_limit
    centers = np.random.rand(3)
    radii = (np.random.rand(3)+0.25) * 2  # between [0.5, 2.5m]
    return ThreeDCircularTraj_fixed(centers, radii, frequencies), np.array([frequencies, centers, radii])


class TwoDLissajous_fixed(TwoDLissajous):
    def __init__(self, A=1, B=1, a=1, b=1, delta=0, height=0, yaw_bool=False):
        super().__init__(A, B, a, b, delta, height, yaw_bool)
        self.t_shift = 0
        
    def shift_forward(self, t_shift):
        self.t_shift += t_shift
        
    def reset(self):
        self.t_shift = 0
        
    def update(self, t):
        output = super().update(t+self.t_shift)
        output["yaw_ddot"] = 0
        return output
        
    
class ThreeDCircularTraj_fixed(object):
    """

    """
    def __init__(self, center=np.array([0,0,0]), radius=np.array([1,1,1]), freq=np.array([0.2,0.2,0.2]), yaw_bool=False):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission.

        Inputs:
            center, the center of the circle (m)
            radius, the radius of the circle (m)
            freq, the frequency with which a circle is completed (Hz)
        """

        self.center = center
        self.cx, self.cy, self.cz = center[0], center[1], center[2]
        self.radius = radius
        self.freq = freq

        self.omega = 2*np.pi*self.freq

        self.yaw_bool = yaw_bool
        self.t_shift = 0

    def shift_forward(self, t_shift):
        self.t_shift += t_shift

    def reset(self):
        self.t_shift = 0

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        if self.t_shift is not None:
            t = t + self.t_shift
        x        = np.array([self.cx + self.radius[0]*np.cos(self.omega[0]*t),
                             self.cy + self.radius[1]*np.sin(self.omega[1]*t),
                             self.cz + self.radius[2]*np.sin(self.omega[2]*t)])
        x_dot    = np.array([-self.radius[0]*self.omega[0]*np.sin(self.omega[0]*t),
                             self.radius[1]*self.omega[1]*np.cos(self.omega[1]*t),
                             self.radius[2]*self.omega[2]*np.cos(self.omega[2]*t)])
        x_ddot   = np.array([-self.radius[0]*(self.omega[0]**2)*np.cos(self.omega[0]*t),
                             -self.radius[1]*(self.omega[1]**2)*np.sin(self.omega[1]*t),
                             -self.radius[2]*(self.omega[2]**2)*np.sin(self.omega[2]*t)])
        x_dddot  = np.array([ self.radius[0]*(self.omega[0]**3)*np.sin(self.omega[0]*t),
                              -self.radius[1]*(self.omega[1]**3)*np.cos(self.omega[1]*t),
                              self.radius[2]*(self.omega[2]**3)*np.cos(self.omega[2]*t)])
        x_ddddot = np.array([self.radius[0]*(self.omega[0]**4)*np.cos(self.omega[0]*t),
                             self.radius[1]*(self.omega[1]**4)*np.sin(self.omega[1]*t),
                             self.radius[2]*(self.omega[2]**4)*np.sin(self.omega[2]*t)])

        # TODO(hersh): these values are suspect - should incorporate frequency, no?
        if self.yaw_bool:
            yaw = 0.8*np.pi/2*np.sin(2.5*t)
            yaw_dot = 0.8*2.5*np.pi/2*np.cos(2.5*t)
            yaw_ddot = 0.8*2.5*2.5*np.pi/2*np.cos(2.5*t)
        else:
            yaw = 0
            yaw_dot = 0
            yaw_ddot = 0

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot, 'yaw_ddot': yaw_ddot}
        return flat_output

def sample_trajectory_static(traj_obj, t_f, dt):
    num_samples = int(t_f/dt)
    state_dim = 3 + 3 + 1 + 1
    sampled_traj = np.zeros((num_samples, state_dim))
    for t in np.arange(0, t_f, dt):
        traj_val = traj_obj.update(t)
        sampled_traj[int(t/dt)][0:3] = traj_val["x"]
        sampled_traj[int(t/dt)][3:6] = traj_val["x_dot"]
        sampled_traj[int(t/dt)][6] = traj_val["yaw"]
        sampled_traj[int(t/dt)][7] = traj_val["yaw_dot"]
    return sampled_traj


