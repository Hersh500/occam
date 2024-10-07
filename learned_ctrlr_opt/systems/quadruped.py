from dataclasses import dataclass
from mpc_controller import a1_sim
from mpc_controller import com_velocity_estimator
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import locomotion_controller
from mpc_controller import openloop_gait_generator
from mpc_controller import raibert_swing_leg_controller
from mpc_controller import torque_stance_leg_controller
import pybullet as pb
import pybullet_data as pd
from pybullet_utils import bullet_client
from typing import Any, Union
import numpy as np
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler
from systems.robots import Robot
from bo_wrappers.botorch_utils import OrnsteinUhlenbeckActionNoise, denormalize
import os
import matplotlib.pyplot as plt

@dataclass
class QuadrupedMPCParams:
    q_roll: float = 5
    q_pitch: float = 5
    q_yaw: float = 0.2
    q_z: float = 10
    q_yaw_vel: float = 1
    q_x_vel: float = 1
    q_y_vel: float = 1
    mpc_body_height: float = 0.24

    def get_q_as_list(self):
        return [self.q_roll,
                self.q_pitch,
                self.q_yaw,
                self.q_z,
                self.q_yaw_vel,
                self.q_x_vel,
                self.q_y_vel]

    def get_full_q_as_list(self):
        return [self.q_roll,
                self.q_pitch,
                self.q_yaw,
                0, 0,
                self.q_z,
                0, 0,
                self.q_yaw_vel,
                self.q_x_vel,
                self.q_y_vel,
                0, 0]

    def get_list(self):
        return [self.q_roll,
                self.q_pitch,
                self.q_yaw,
                self.q_z,
                self.q_yaw_vel,
                self.q_x_vel,
                self.q_y_vel,
                self.mpc_body_height]

    @staticmethod
    def get_bounds():
        return np.array([[0, 15],
                        [0, 15],
                        [0, 15],
                        [0, 15],
                        [0, 15],
                        [0, 15],
                        [0, 15],
                        [0.17, 0.4]])

    @staticmethod
    def get_names():
        return ["q_roll",
                "q_pitch",
                "q_yaw",
                "q_z",
                "q_yaw_vel",
                "q_x_vel",
                "q_y_vel",
                "body_height"]

    @staticmethod
    def get_num():
        return 8

    @staticmethod
    def generate_random(to_randomize=None):
        if to_randomize is None:
            to_randomize = [i for i in range(QuadrupedMPCParams.get_num())]
        params = QuadrupedMPCParams().get_list()
        for idx in to_randomize:
            b = QuadrupedMPCParams.get_bounds()[idx]
            params[idx] = np.random.uniform(b[0], b[1])
        return QuadrupedMPCParams(*params)


@dataclass
class QuadrupedPDParams:
    kp: float = 0.1
    kd: float = 0.2


@dataclass
class QuadrupedGaitParams:
    stance_duration_1: float = 0.3
    stance_duration_2: float = 0.3
    stance_duration_3: float = 0.3
    stance_duration_4: float = 0.3
    duty_factor_1: float = 0.6
    duty_factor_2: float = 0.6
    duty_factor_3: float = 0.6
    duty_factor_4: float = 0.6

    def get_stance_duration_list(self):
        return [self.stance_duration_1,
                self.stance_duration_2,
                self.stance_duration_3,
                self.stance_duration_4]

    def get_duty_factor_list(self):
        return [self.duty_factor_1,
                self.duty_factor_2,
                self.duty_factor_3,
                self.duty_factor_4]


@dataclass
class QuadrupedModelParams:
    base_mass: float = 11 # kg
    base_ixx: float = 0.0017
    base_iyy: float = 0.0057
    base_izz: float = 0.0064
    friction: float = 0.8

    def get_list(self):
        return [self.base_mass,
                self.base_ixx,
                self.base_iyy,
                self.base_izz,
                self.friction]

    def get_full_inertia_tuple(self):
        return (self.base_ixx, 0, 0,
                0, self.base_iyy, 0,
                0, 0, self.base_izz)

    @staticmethod
    def generate_random_inertia():
        inertia_bounds = QuadrupedModelParams.get_bounds()[1:4,:]
        while True:
            inertias = np.random.uniform(inertia_bounds[:,0], inertia_bounds[:,1])
            if inertias[0] + inertias[1] > inertias[2]:
                return inertias

    @staticmethod
    def get_names():
        return ["base_mass",
                "base_ixx",
                "base_iyy",
                "base_izz",
                "friction"]

    @staticmethod
    def get_bounds():
        return np.array([[8, 14],
                         [0.001, 0.004],
                         [0.002, 0.01],
                         [0.003, 0.01],
                         [0.5, 1.1]])

    @staticmethod
    def get_num():
        return 5

    @staticmethod
    def generate_random(to_randomize=None):
        if to_randomize is None:
            to_randomize = [i for i in range(QuadrupedModelParams.get_num())]
        params = QuadrupedModelParams().get_list()
        bounds = QuadrupedModelParams.get_bounds()
        for idx in to_randomize:
            params[idx] = np.random.uniform(bounds[idx][0], bounds[idx][1])

        # check validity of inertia tensor
        while params[1] + params[2] < params[3]:
            for i in range(1, 4):
                if i in to_randomize:
                    params[i] = np.random.uniform(bounds[i][0], bounds[i][1])
        return QuadrupedModelParams(*params)


def change_robot_params(robot_sim: a1_sim.SimpleRobot,
                        new_params: QuadrupedModelParams):
    robot_sim.pybullet_client.changeDynamics(robot_sim.quadruped, -1, mass=new_params.base_mass)
    robot_sim.pybullet_client.changeDynamics(robot_sim.quadruped,
                                             -1,
                                             localInertiaDiagonal=[new_params.base_ixx,
                                                                   new_params.base_iyy,
                                                                   new_params.base_izz])
    foot_links = robot_sim.GetFootLinkIDs()
    for fl in foot_links:
        robot_sim.pybullet_client.changeDynamics(robot_sim.quadruped,
                                                 fl,
                                                 lateralFriction=new_params.friction)


# Since the parameters of the true robot are unknown at test time, this sets up an MPC
# that uses nominal parameters instead
def setup_nominal_swst_ctrlr(ctrlr_params: QuadrupedMPCParams,
                             robot,
                             gait_generator,
                             state_estimator,
                             desired_speed,
                             desired_twisting_speed):
    nominal_params = QuadrupedModelParams()
    st_controller = torque_stance_leg_controller.TorqueStanceLegController(
        robot,
        gait_generator,
        state_estimator,
        desired_speed=desired_speed,
        desired_twisting_speed=desired_twisting_speed,
        desired_body_height=ctrlr_params.mpc_body_height,
        body_mass=nominal_params.base_mass,
        body_inertia=nominal_params.get_full_inertia_tuple(),
        mpc_weights=ctrlr_params.get_full_q_as_list())

    sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
        robot,
        gait_generator,
        state_estimator,
        desired_speed=desired_speed,
        desired_twisting_speed=desired_twisting_speed,
        desired_height=ctrlr_params.mpc_body_height,
        foot_clearance=0.01)
    return st_controller, sw_controller


def setup_gait_generator(gait_params: QuadrupedGaitParams,
                         robot):
    _INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]
    _MAX_TIME_SECONDS = 50

    _INIT_LEG_STATE = (
        gait_generator_lib.LegState.SWING,
        gait_generator_lib.LegState.STANCE,
        gait_generator_lib.LegState.STANCE,
        gait_generator_lib.LegState.SWING,
    )
    return openloop_gait_generator.OpenloopGaitGenerator(
        robot,
        stance_duration=gait_params.get_stance_duration_list(),
        duty_factor=gait_params.get_duty_factor_list(),
        initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
        initial_leg_state=_INIT_LEG_STATE)


def setup_controller(robot: Any,
                     gait_params: QuadrupedGaitParams,
                     mpc_ctrlr_params: QuadrupedMPCParams,
                     desired_speed,
                     desired_twisting_speed):
    state_estimator = com_velocity_estimator.COMVelocityEstimator(robot,
                                                                  window_size=20)
    gait_gen = setup_gait_generator(gait_params, robot)
    st_ctrlr, sw_ctrlr = setup_nominal_swst_ctrlr(mpc_ctrlr_params, robot, gait_gen, state_estimator,
                                                  desired_speed, desired_twisting_speed)
    controller = locomotion_controller.LocomotionController(
        robot=robot,
        gait_generator=gait_gen,
        state_estimator=state_estimator,
        swing_leg_controller=sw_ctrlr,
        stance_leg_controller=st_ctrlr,
        clock=robot.GetTimeSinceReset)
    return controller


def setup_physics(p, render):
    if not render:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    else:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    p.setAdditionalSearchPath(pd.getDataPath())

    num_bullet_solver_iterations = 30

    p.setPhysicsEngineParameter(numSolverIterations=num_bullet_solver_iterations)

    p.setPhysicsEngineParameter(enableConeFriction=0)
    p.setPhysicsEngineParameter(numSolverIterations=30)
    simulation_time_step = 0.001

    p.setTimeStep(simulation_time_step)

    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(enableConeFriction=0)
    p.setAdditionalSearchPath(pd.getDataPath())
    p.resetDebugVisualizerCamera(cameraDistance=5.0, cameraYaw=50.0, cameraPitch=-35.0,
                                 cameraTargetPosition=(0.0, 0.0, 0.0))


def _update_controller_params(controller, lin_speed, ang_speed):
    controller.swing_leg_controller.desired_speed = lin_speed
    controller.swing_leg_controller.desired_twisting_speed = ang_speed
    controller.stance_leg_controller.desired_speed = lin_speed
    controller.stance_leg_controller.desired_twisting_speed = ang_speed


class QuadrupedMPC(Robot):
    def __init__(self, quadruped_params: QuadrupedModelParams = QuadrupedModelParams(),
                 gains_to_optimize: list = [],
                 render=False):
        self.params = quadruped_params
        self.gains_to_optimize = gains_to_optimize
        self.render = render
        self.prior_dir = "priors/quadruped_mpc"
        self.test_set_dir = "test_sets/quadruped_mpc"
        self.ParamsT = QuadrupedModelParams
        self.ControllerParamsT = QuadrupedMPCParams

    def evaluate_x(self, x: Union[list, np.array], render_override=False):
        if self.render or render_override:
            p = bullet_client.BulletClient(connection_mode=pb.GUI)
        else:
            p = bullet_client.BulletClient(connection_mode=pb.DIRECT)
        setup_physics(p, self.render or render_override)
        p.loadURDF("plane.urdf")
        # print(p.getDebugVisualizerCamera())
        robot_id = p.loadURDF(a1_sim.URDF_NAME, a1_sim.START_POS)
        robot = a1_sim.SimpleRobot(p, robot_id, simulation_time_step=0.001)
        change_robot_params(robot, self.params)
        gait_params = QuadrupedGaitParams()
        mpc_params_list = np.array(QuadrupedMPCParams().get_list())
        mpc_params_list[self.gains_to_optimize] = x
        desired_speed = [0.5, 0]
        desired_twist_speed = 0
        controller = setup_controller(robot, gait_params, QuadrupedMPCParams(*mpc_params_list),
                                      desired_speed, desired_twist_speed)
        controller.reset()
        max_time = 5
        current_time = robot.GetTimeSinceReset()

        avg_speed_error = 0
        avg_body_rp = 0
        avg_effort = 0
        ctr = 0
        while current_time < max_time:
            # pos,orn = p.getBasePositionAndOrientation(robot_uid)
            # print("pos=",pos, " orn=",orn)
            p.submitProfileTiming("loop")

            # Updates the controller behavior parameters.
            # lin_speed, ang_speed = (0., 0., 0.), 0.
            # _update_controller_params(controller, (0, 0), 0)

            # Needed before every call to get_action().
            controller.update()
            hybrid_action, info = controller.get_action()

            robot.Step(hybrid_action)
            # Accessing a protected member here, but that's fine.
            avg_effort += np.sum(robot._observed_motor_torques)
            current_speed = robot.GetBaseVelocity()
            avg_speed_error += np.linalg.norm(np.array(current_speed)[:2] - np.array(desired_speed))
            avg_body_rp += np.sum(np.abs(np.array(robot.GetBaseRollPitchYaw()[:2])))
            current_time = robot.GetTimeSinceReset()
            ctr += 1
            p.submitProfileTiming()
        p.disconnect()
        return np.array([avg_speed_error, avg_effort, avg_body_rp])/ctr

    @staticmethod
    def worker(gains_to_optimize, params_to_sweep, seed):
        np.random.seed(seed)
        thetas = QuadrupedModelParams.generate_random(params_to_sweep)
        robot = QuadrupedMPC(thetas, gains_to_optimize, render=False)
        # This syntax is a little clunky here.
        gains = np.array(QuadrupedMPCParams.generate_random(gains_to_optimize).get_list())
        y = robot.evaluate_x(gains[gains_to_optimize])
        x = np.append(np.array(gains), np.array(thetas.get_list()))
        print(y)
        return x, y

    @staticmethod
    def gather_data(n_points, gains_to_optimize, params_to_sweep):
        print(f"n_points = {n_points}")
        num_done = 0
        X = np.zeros((n_points, QuadrupedMPCParams.get_num() + QuadrupedModelParams.get_num()))
        y = np.zeros((n_points, 3))
        while num_done < n_points:
            print(f"num_done = {num_done}")
            num_proc = max(1, mp.cpu_count() - 4)
            p = mp.Pool(processes=num_proc)
            return_val_array = p.starmap(QuadrupedMPC.worker,
                                         [(gains_to_optimize, params_to_sweep, np.random.randint(0, 10000)) for j in range(num_proc)])
            for i in range(num_proc):
                X[num_done+i,:] = return_val_array[i][0]
                y[num_done+i, :] = return_val_array[i][1]
            num_done += num_proc
        scaler = StandardScaler().fit(y)
        return X, y, scaler

    def get_thetas(self):
        return np.array(self.params.get_list())

    def perf_metric_names(self):
        return np.array(["Avg speed error", "avg effort", "avg body angle"])

    def get_gain_bounds(self):
        return QuadrupedMPCParams.get_bounds()[self.gains_to_optimize]

    def get_theta_bounds(self):
        return QuadrupedModelParams.get_bounds()

    def get_gain_dim(self):
        return len(self.gains_to_optimize)

    def get_gain_names(self):
        return np.array(QuadrupedMPCParams.get_names())

    def get_nominal_values(self):
        return np.array(QuadrupedMPCParams().get_list() + QuadrupedModelParams().get_list())

# Tracks a trajectory specified through velocity commands.
# com_velocity_traj = [[x_vel, y_vel, yaw_vel]]
# What should evaluate_x return? low-level information, "Trajectory descriptors", etc?
# ie. what should the latent space be?
class QuadrupedMPCVelTraj(Robot):
    def __init__(self, quadruped_params: QuadrupedModelParams = QuadrupedModelParams(),
                 gains_to_optimize: list = [],
                 com_velocity_trajectory: list = [],
                 traj_dt=1.0,
                 render=False):
        self.params = quadruped_params
        self.gains_to_optimize = gains_to_optimize
        self.render = render
        self.prior_dir = "priors/quadruped_mpc"
        self.test_set_dir = "test_sets/quadruped_mpc"
        self.ParamsT = QuadrupedModelParams
        self.ControllerParamsT = QuadrupedMPCParams
        self.com_traj = np.array(com_velocity_trajectory)
        self.traj_dt = traj_dt

    def get_commands(self):
        return self.com_traj


    def evaluate_x(self,
                   x: Union[list, np.array],
                   render_override=False,
                   record_video=False,
                   video_path="test.mp4"):
        if self.render or render_override:
            p = bullet_client.BulletClient(connection_mode=pb.GUI)
        else:
            p = bullet_client.BulletClient(connection_mode=pb.DIRECT)
        if record_video and render_override or self.render:
            if not os.path.exists(os.path.dirname(video_path)):
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_path)
        setup_physics(p, self.render or render_override)
        p.loadURDF("plane.urdf")
        # print(p.getDebugVisualizerCamera())
        robot_id = p.loadURDF(a1_sim.URDF_NAME, a1_sim.START_POS)
        robot = a1_sim.SimpleRobot(p, robot_id, simulation_time_step=0.001)
        change_robot_params(robot, self.params)
        gait_params = QuadrupedGaitParams()
        mpc_params_list = np.array(QuadrupedMPCParams().get_list())
        mpc_params_list[self.gains_to_optimize] = x
        desired_speed = self.com_traj[0][:3]
        desired_twist_speed = self.com_traj[0][3]
        controller = setup_controller(robot, gait_params, QuadrupedMPCParams(*mpc_params_list),
                                      desired_speed, desired_twist_speed)
        controller.reset()
        max_time = self.com_traj.shape[0] * self.traj_dt
        # print(f"max time is {max_time}")
        current_time = robot.GetTimeSinceReset()

        avg_speed_error = 0
        avg_body_rp = 0
        avg_effort = 0
        avg_yaw_vel_error = 0
        ctr = 0
        while current_time < max_time:
            # pos,orn = p.getBasePositionAndOrientation(robot_uid)
            # print("pos=",pos, " orn=",orn)
            p.submitProfileTiming("loop")

            # Updates the controller behavior parameters.
            # lin_speed, ang_speed = (0., 0., 0.), 0.
            # _update_controller_params(controller, (0, 0), 0)
            current_vel_des = self.com_traj[int(np.floor(current_time/self.traj_dt))]

            # Needed before every call to get_action().
            controller.update_commands(desired_com_velocity=current_vel_des[:3],
                                       desired_yaw_vel=current_vel_des[3])
            controller.update()
            hybrid_action, info = controller.get_action()

            robot.Step(hybrid_action)
            # Accessing a protected member here, but that's fine.
            avg_effort += np.sum(np.abs(robot._observed_motor_torques))

            # NEED TO CONVERT THESE TO BODY FRAME...COMMANDED VELOCITIES ARE CURRENTLY IN BODY FRAME
            current_speed_world = robot.GetBaseVelocity()
            yaw = robot.GetBaseRollPitchYaw()[2]
            current_speed_body = np.array([current_speed_world[0]*np.cos(yaw) + current_speed_world[1]*np.sin(yaw),
                                           -current_speed_world[0]*np.sin(yaw) + current_speed_world[1]*np.cos(yaw)])
            if render_override:
                if ctr % 50 == 0:
                    print(f"Current body frame speed = {current_speed_body}, cmd speed is {current_vel_des[:3]}, current time is {current_time}")
                    print(f"Current yaw vel is {robot.GetBaseRollPitchYawRate()[2]}, cmd is {current_vel_des[3]}")
            avg_speed_error += np.linalg.norm(np.array(current_speed_body)[:2] - current_vel_des[:2])
            avg_body_rp += np.sum(np.abs(np.array(robot.GetBaseRollPitchYaw()[:2])))
            yaw_vel = robot.GetBaseRollPitchYawRate()[2]
            avg_yaw_vel_error += np.linalg.norm(yaw_vel - current_vel_des[3])
            current_time = robot.GetTimeSinceReset()
            ctr += 1
            p.submitProfileTiming()
        p.disconnect()
        return np.array([avg_speed_error, avg_yaw_vel_error, avg_effort, avg_body_rp])/ctr

    @staticmethod
    def worker(gains_to_optimize, params_to_sweep, seed):
        print(f"NOT WORKING YET!")
        np.random.seed(seed)
        thetas = QuadrupedModelParams.generate_random(params_to_sweep)
        robot = QuadrupedMPC(thetas, gains_to_optimize, render=False)
        # This syntax is a little clunky here.
        gains = np.array(QuadrupedMPCParams.generate_random(gains_to_optimize).get_list())
        y = robot.evaluate_x(gains[gains_to_optimize])
        x = np.append(np.array(gains), np.array(thetas.get_list()))
        print(y)
        return x, y

    @staticmethod
    def gather_data(n_points, gains_to_optimize, params_to_sweep):
        print(f"n_points = {n_points}")
        num_done = 0
        X = np.zeros((n_points, QuadrupedMPCParams.get_num() + QuadrupedModelParams.get_num()))
        y = np.zeros((n_points, 3))
        while num_done < n_points:
            print(f"num_done = {num_done}")
            num_proc = max(1, mp.cpu_count() - 4)
            p = mp.Pool(processes=num_proc)
            return_val_array = p.starmap(QuadrupedMPC.worker,
                                         [(gains_to_optimize, params_to_sweep, np.random.randint(0, 10000)) for j in range(num_proc)])
            for i in range(num_proc):
                X[num_done+i,:] = return_val_array[i][0]
                y[num_done+i, :] = return_val_array[i][1]
            num_done += num_proc
        scaler = StandardScaler().fit(y)
        return X, y, scaler

    def get_thetas(self):
        return np.array(self.params.get_list())

    def perf_metric_names(self):
        return np.array(["Avg speed error", "avg effort", "avg body angle"])

    def get_gain_bounds(self):
        return QuadrupedMPCParams.get_bounds()[self.gains_to_optimize]

    def get_theta_bounds(self):
        return QuadrupedModelParams.get_bounds()

    def get_gain_dim(self):
        return len(self.gains_to_optimize)

    def get_gain_names(self):
        return np.array(QuadrupedMPCParams.get_names())

    def get_nominal_values(self):
        return np.array(QuadrupedMPCParams().get_list() + QuadrupedModelParams().get_list())

    @staticmethod
    def generate_random_vel_commands_ou(seed, gen_params, cmd_bounds, num_commands):
        np.random.seed(seed)
        noise_policy = OrnsteinUhlenbeckActionNoise(mu=gen_params["mu"],
                                                    sigma=gen_params["sigma"],
                                                    theta=gen_params["theta"])
        noise_policy.reset()
        trajectory = np.zeros((num_commands, 4))
        for i in range(num_commands):
            action_norm = np.clip(noise_policy(), -1, 1)
            action = (action_norm + 1) / 2 * (cmd_bounds[:, 1] - cmd_bounds[:, 0]) + cmd_bounds[:, 0]
            trajectory[i,0:2] = action
        return trajectory

    @staticmethod
    def generate_random_vel_commands(seed, cmd_bounds, num_commands, norm_std=0.1):
        np.random.seed(seed)
        trajectory = np.zeros((num_commands, 4))
        trajectory[0,[0,3]] = np.random.rand(2)
        for i in range(1, num_commands):
            trajectory[i,[0,3]] = np.clip(trajectory[i-1,[0,3]] + (np.random.rand(2)-0.5)*norm_std,
                                        0, 1)
        trajectory[:,[0,3]] = denormalize(trajectory[:,[0,3]], cmd_bounds)
        return trajectory

    @staticmethod
    def plot_command_trajectory(com_traj, traj_dt, ax):
        sim_dt = 0.01
        states = np.zeros((int(traj_dt * com_traj.shape[0]/sim_dt), 3))
        current_time = 0
        for i in range(1, states.shape[0]):
            current_vel_des = com_traj[int(np.floor(current_time / traj_dt))]
            states[i][0] += states[i-1][0] + np.cos(states[i-1][2])*current_vel_des[0]*sim_dt - np.sin(states[i-1][2])*current_vel_des[1]*sim_dt
            states[i][1] += states[i-1][1] + np.sin(states[i-1][2])*current_vel_des[0]*sim_dt + np.cos(states[i-1][2])*current_vel_des[1]*sim_dt
            states[i][2] += states[i-1][2] + current_vel_des[3]*sim_dt
            current_time += sim_dt
        ax.scatter(states[:,0], states[:,1])
        return ax

class QuadrupedLowLevelState:
    def __init__(self, quadruped_params: QuadrupedModelParams = QuadrupedModelParams(),
                 render=False):
        self.params = quadruped_params
        self.render = render
        self.ParamsT = QuadrupedModelParams
        self.ControllerParamsT = QuadrupedMPCParams

    # returns low-level states and torques, along with the aggregate metrics
    def evaluate_x(self, x: Union[list, np.array], render_override=False):
        if self.render or render_override:
            p = bullet_client.BulletClient(connection_mode=pb.GUI)
        else:
            p = bullet_client.BulletClient(connection_mode=pb.DIRECT)
        setup_physics(p, self.render or render_override)
        p.loadURDF("plane.urdf")
        # print(p.getDebugVisualizerCamera())
        robot_id = p.loadURDF(a1_sim.URDF_NAME, a1_sim.START_POS)
        robot = a1_sim.SimpleRobot(p, robot_id, simulation_time_step=0.001)
        change_robot_params(robot, self.params)
        gait_params = QuadrupedGaitParams()
        mpc_params_list = np.array(QuadrupedMPCParams().get_list())
        mpc_params_list[self.gains_to_optimize] = x
        desired_speed = [0.5, 0]
        desired_twist_speed = 0
        controller = setup_controller(robot, gait_params, QuadrupedMPCParams(*mpc_params_list),
                                      desired_speed, desired_twist_speed)
        controller.reset()
        max_time = 5
        current_time = robot.GetTimeSinceReset()

        avg_speed_error = 0
        avg_body_rp = 0
        avg_effort = 0
        ctr = 0
        states = []
        torques = []  # are torques too low-level? Can I get forces instead?
        while current_time < max_time:
            # pos,orn = p.getBasePositionAndOrientation(robot_uid)
            # print("pos=",pos, " orn=",orn)
            p.submitProfileTiming("loop")

            # Updates the controller behavior parameters.
            # lin_speed, ang_speed = (0., 0., 0.), 0.
            # _update_controller_params(controller, (0, 0), 0)

            # Needed before every call to get_action().
            controller.update()
            hybrid_action, info = controller.get_action()
            torques.append(hybrid_action)
            states.append(robot.GetBaseRollPitchYaw() + robot.GetBaseRollPitchYawRate() + robot.GetBaseVelocity())

            robot.Step(hybrid_action)
            # Accessing a protected member here, but that's fine.
            avg_effort += np.sum(robot._observed_motor_torques)
            current_speed = robot.GetBaseVelocity()
            avg_speed_error += np.linalg.norm(np.array(current_speed)[:2] - np.array(desired_speed))
            avg_body_rp += np.sum(np.abs(np.array(robot.GetBaseRollPitchYaw()[:2])))
            current_time = robot.GetTimeSinceReset()
            ctr += 1
            p.submitProfileTiming()
        p.disconnect()
        return states, torques, np.array([avg_speed_error, avg_effort, avg_body_rp])/ctr

    @staticmethod
    def worker(gains_to_optimize, params_to_sweep, seed):
        np.random.seed(seed)
        thetas = QuadrupedModelParams.generate_random(params_to_sweep)
        robot = QuadrupedMPC(thetas, gains_to_optimize, render=False)
        # This syntax is a little clunky here.
        gains = np.array(QuadrupedMPCParams.generate_random(gains_to_optimize).get_list())
        y = robot.evaluate_x(gains[gains_to_optimize])
        x = np.append(np.array(gains), np.array(thetas.get_list()))
        print(y)
        return x, y

    @staticmethod
    def gather_data(n_points, gains_to_optimize, params_to_sweep):
        print(f"n_points = {n_points}")
        num_done = 0
        X = np.zeros((n_points, QuadrupedMPCParams.get_num() + QuadrupedModelParams.get_num()))
        y = np.zeros((n_points, 3))
        while num_done < n_points:
            print(f"num_done = {num_done}")
            num_proc = max(1, mp.cpu_count() - 4)
            p = mp.Pool(processes=num_proc)
            return_val_array = p.starmap(QuadrupedMPC.worker,
                                         [(gains_to_optimize, params_to_sweep, np.random.randint(0, 10000)) for j in range(num_proc)])
            for i in range(num_proc):
                X[num_done+i,:] = return_val_array[i][0]
                y[num_done+i, :] = return_val_array[i][1]
            num_done += num_proc
        scaler = StandardScaler().fit(y)
        return X, y, scaler

    def get_thetas(self):
        return np.array(self.params.get_list())

    def perf_metric_names(self):
        return np.array(["Avg speed error", "avg effort", "avg body angle"])

    def get_gain_bounds(self):
        return QuadrupedMPCParams.get_bounds()[self.gains_to_optimize]

    def get_theta_bounds(self):
        return QuadrupedModelParams.get_bounds()

    def get_gain_dim(self):
        return len(self.gains_to_optimize)

    def get_gain_names(self):
        return np.array(QuadrupedMPCParams.get_names())

    def get_nominal_values(self):
        return np.array(QuadrupedMPCParams().get_list() + QuadrupedModelParams().get_list())

def main():
    model_params = QuadrupedModelParams()
    gains_to_optimize = [i for i in range(8)]
    robot = QuadrupedMPC(model_params, gains_to_optimize, False)
    metrics = robot.evaluate_x(QuadrupedMPCParams().get_list())
    print(metrics)
