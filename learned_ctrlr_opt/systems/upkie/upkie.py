from __future__ import annotations
import os
from dataclasses import dataclass
from datetime import datetime
import gymnasium as gym
import numpy as np
from typing import Optional, SupportsFloat, Any
import h5py
import multiprocessing as mp


from upkie.envs.wrappers import LowPassFilterAction
from gymnasium.core import ActType, ObsType, RenderFrame
from qpmpc import MPCQP, Plan, solve_mpc
from qpmpc.systems import WheeledInvertedPendulum
from qpmpc.mpc_problem import MPCProblem

from learned_ctrlr_opt.utils.dataset_utils import denormalize
from learned_ctrlr_opt.systems.upkie.proxqp_workspace import ProxQPWorkspace


@dataclass
class WheeledInvertedPendulumParams:
    leg_length: float = 0.58
    wheel_radius: float = 0.06
    action_lpf: float = 0.07
    mpc_sampling_period: float = 0.1

    @staticmethod
    def get_bounds():
        return np.array([[0.4, 0.6],
                         [0.04, 0.08],
                         [0.00, 0.15],
                         [0.05, 0.15]])

    @classmethod
    def generate_random(cls, params_to_randomize):
        random_params = denormalize(np.random.rand(4), cls.get_bounds())
        params = cls().get_list()
        params[params_to_randomize] = random_params[params_to_randomize]
        return cls(*list(params))

    def get_list(self):
        return np.array([self.leg_length, self.wheel_radius, self.action_lpf,
                         self.mpc_sampling_period])


@dataclass
class MPCBalancerParams:
    leg_length: float = 0.58
    wheel_radius: float = 0.06

    @staticmethod
    def get_bounds():
        return np.array([[0.3, 0.7],
                        [0.04, 0.08]])

    def get_list(self):
        return np.array([self.leg_length,
                         self.wheel_radius])

    @classmethod
    def generate_random(cls, params_to_randomize):
        random_params = denormalize(np.random.rand(2), cls.get_bounds())
        params = cls().get_list()
        params[params_to_randomize] = random_params[params_to_randomize]
        return cls(*list(params))


class WheeledInvertedPendulumEnv(gym.Env):
    def __init__(self,
                 leg_length: float,
                 wheel_radius: float,
                 sampling_period: float,
                 crash_angle: float):
        self.leg_length = leg_length
        self.wheel_radius = wheel_radius
        self.dt = sampling_period
        self.gravity = 9.81
        self.crash_angle = crash_angle

        self.omega = np.sqrt(self.gravity / self.leg_length)

        T = self.dt
        omega = self.omega
        g = self.gravity

        self.A_disc = np.array(
            [
                [1.0, 0.0, T, 0.0],
                [0.0, np.cosh(T * omega), 0.0, np.sinh(T * omega) / omega],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, omega * np.sinh(T * omega), 0.0, np.cosh(T * omega)],
            ]
        )

        self.B_disc = np.array(
            [
                [T ** 2 / 2.0],
                [-np.cosh(T * omega) / g + 1.0 / g],
                [T],
                [-omega * np.sinh(T * omega) / g],
            ]
        )

        self.initial_state = np.zeros(4)
        self.state = self.initial_state

        MAX_BASE_PITCH: float = np.pi
        MAX_GROUND_POSITION: float = float("inf")
        MAX_BASE_ANGULAR_VELOCITY: float = 1000.0  # rad/s
        observation_limit = np.array(
            [
                MAX_GROUND_POSITION,
                MAX_BASE_PITCH,
                1.0,
                MAX_BASE_ANGULAR_VELOCITY,
            ],
            dtype=float,
        )
        self.observation_space = gym.spaces.Box(
            -observation_limit,
            +observation_limit,
            shape=observation_limit.shape,
            dtype=observation_limit.dtype,
        )

        # gymnasium.Env: action_space
        action_limit = np.array([1.0], dtype=float)
        self.action_space = gym.spaces.Box(
            -action_limit,
            +action_limit,
            shape=action_limit.shape,
            dtype=action_limit.dtype,
        )

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # since our input is the angular acceleration, multiply by the wheel radius
        # one step of the dynamics
        # next_state = np.dot(self.A_disc, self.state) + np.dot(self.B_disc, action * self.wheel_radius)
        next_state = self.A_disc @ self.state + self.B_disc @ (action * self.wheel_radius)

        # check the state for failure
        terminated = np.abs(next_state[1]) > self.crash_angle
        truncated = False

        self.state = next_state
        return next_state, 0, terminated, truncated, {}

    def reset(
        self,
        initial_state = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ):
        if initial_state is not None:
            self.state = initial_state
        else:
            self.state = self.initial_state

        return self.state

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return None


# Implements Wheeled Inverted Pendulum with Angular Acceleration as the Input
class WheeledInvertedPendulumAngularAccMPC(WheeledInvertedPendulum):
    wheel_radius: float

    def __init__(self, length: float = 0.6,
                 max_ground_accel: float = 10.0,
                 nb_timesteps: int = 12,
                 sampling_period: float = 0.1,
                 wheel_radius: float = 0.06):
        super().__init__(length, max_ground_accel, nb_timesteps, sampling_period)
        self.wheel_radius = wheel_radius

        # when simulating the actual system, simulates a low pass filter over the actions
        # number of taps is the size of the moving average
        # self.action_filter_alpha = action_filter_alpha
        # self.previous_action = np.zeros(1)

    def build_mpc_problem(
        self,
        stage_input_cost_weight: float = 1e-3,
        stage_state_cost_weight: Optional[float] = None,
        terminal_cost_weight: Optional[float] = 1.0,
    ) -> MPCProblem:
        T = self.sampling_period
        omega = self.omega
        g = self.GRAVITY

        A_disc = np.array(
            [
                [1.0, 0.0, T, 0.0],
                [0.0, np.cosh(T * omega), 0.0, np.sinh(T * omega) / omega],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, omega * np.sinh(T * omega), 0.0, np.cosh(T * omega)],
            ]
        )

        # since our input is the angular acceleration, we multiply by the wheel radius
        B_disc = np.array(
            [
                [T ** 2 / 2.0],
                [-np.cosh(T * omega) / g + 1.0 / g],
                [T],
                [-omega * np.sinh(T * omega) / g],
            ]
        ) * self.wheel_radius

        ground_accel_ineq_matrix = np.vstack([np.eye(1), -np.eye(1)]) / self.wheel_radius
        ground_accel_ineq_vector = np.hstack(
            [
                self.max_ground_accel,
                self.max_ground_accel,
            ]
        ) / self.wheel_radius

        return MPCProblem(
            transition_state_matrix=A_disc,
            transition_input_matrix=B_disc,
            ineq_state_matrix=None,
            ineq_input_matrix=ground_accel_ineq_matrix,
            ineq_vector=ground_accel_ineq_vector,
            nb_timesteps=self.nb_timesteps,
            terminal_cost_weight=terminal_cost_weight,
            stage_state_cost_weight=stage_state_cost_weight,
            stage_input_cost_weight=stage_input_cost_weight * self.wheel_radius,
            initial_state=None,
            goal_state=None,
            target_states=None,
        )


# Use this to fit with OCCAM's API.
# two environments - one is the "true" environment and one is the "perfect" wheeled env, per email with Stephane
class DelayedUpkieSystem:
    def __init__(self, params: WheeledInvertedPendulumParams,
                 time_horizon_s: float):
        # create upkie system
        self.params = params
        self.time_horizon_s = time_horizon_s
        self.tstep = 0.01

    def evaluate_gain(self, gains, init_state=None, render=False):
        # define the ground truth environment with params
        gt_env = LowPassFilterAction(
            WheeledInvertedPendulumEnv(
                self.params.leg_length,
                self.params.wheel_radius,
                self.tstep,
                1.0
            ),
            time_constant=self.params.action_lpf)
        observation = gt_env.reset(initial_state=init_state)

        # create MPC problem with "gains"
        pendulum = WheeledInvertedPendulumAngularAccMPC(
            length=gains[0],
            wheel_radius=gains[1],
            sampling_period=self.params.mpc_sampling_period
        )
        mpc_problem = pendulum.build_mpc_problem(
            terminal_cost_weight=1.0,
            stage_state_cost_weight=1e-3,
            stage_input_cost_weight=1e-3,
        )
        mpc_problem.initial_state = observation
        mpc_qp = MPCQP(mpc_problem)
        workspace = ProxQPWorkspace(mpc_qp, update_preconditioner=True, verbose=False)
        commanded_accel = 0

        # for n steps, run the MPC algorithm and step the ground truth environment
        action = np.zeros(1)
        num_steps = int(self.time_horizon_s/self.tstep)
        states = np.zeros((num_steps, 4))
        inputs = np.zeros(num_steps)
        nx = 4
        for i in range(num_steps):
            states[i] = observation
            inputs[i] = commanded_accel
            action[0] = commanded_accel
            observation, _, terminated, truncated, info = gt_env.step(action)
            observation_noisy = observation + np.random.randn() * 1e-2
            if render:
                print(f"action = {action}")
                print(f"state = {observation}")
            if terminated:
                print("terminated!")
                break

            initial_state = observation_noisy
            if np.abs(observation[2]) > 1.0:
                print(f"Velocity exceeding 1.0!")
            target_states = np.zeros((pendulum.nb_timesteps + 1) * nx)
            mpc_problem.update_initial_state(initial_state)
            mpc_problem.update_goal_state(target_states[-nx:])
            mpc_problem.update_target_states(target_states[:-nx])

            mpc_qp.update_cost_vector(mpc_problem)
            qpsol = workspace.solve(mpc_qp)
            if not qpsol.found:
                print("No solution found to the MPC problem")

            plan = Plan(mpc_problem, qpsol)
            pendulum.state = observation_noisy
            commanded_accel = plan.first_input[0]
            # print("------------------------")
            # print(f"Plan expected states:")
            # print(plan.states)
            # print("------------------------")
            # print(f"expected next state = {next_state}")

        # compute performance metrics
        velocity_error = np.sum(np.abs(states[:i,2]))/i
        angle_error = np.sum(np.abs(states[:i,1]))/i
        effort = np.sum(np.abs(inputs[:i]))/i
        return np.array([angle_error, velocity_error, effort]), states

    @staticmethod
    def perf_metric_names():
        return ["angle error", "velocity error", "effort"]


def upkie_worker(intrinsic,
                 gains,
                 init_state,
                 ep_length):
    robot = DelayedUpkieSystem(params=WheeledInvertedPendulumParams(*intrinsic), time_horizon_s=ep_length)
    metrics, traj = robot.evaluate_gain(gains, init_state=init_state)
    return metrics


def gather_upkie_balancing_mpc_data(num_batches,
                                    batch_size,
                                    thetas_to_randomize,
                                    high_level_folder,
                                    init_state_bounds,
                                    ep_length):
    intrinsics = np.zeros((num_batches, len(WheeledInvertedPendulumParams().get_list())))
    gains = np.zeros((num_batches, batch_size, len(MPCBalancerParams().get_list())))
    ref_tracks_enc = np.zeros((num_batches, batch_size, 4))
    metrics = np.zeros((num_batches, batch_size, len(DelayedUpkieSystem.perf_metric_names())))
    for b in range(num_batches):
        print(f"on batch {b}")
        intrinsic = WheeledInvertedPendulumParams.generate_random(thetas_to_randomize).get_list()
        intrinsics[b] = intrinsic
        for i in range(batch_size):
            gains[b,i] = MPCBalancerParams.generate_random([0, 1]).get_list()
            ref_tracks_enc[b,i] = denormalize(np.random.rand(4), init_state_bounds)
            metrics[b,i] = upkie_worker(intrinsics[b],
                                   gains[b,i],
                                   ref_tracks_enc[b,i],
                                   ep_length=ep_length)
        # args = [(intrinsic,
        #          gains[b,i],
        #          ref_tracks_enc[b,i],
        #          ep_length) for i in range(batch_size)]

        # metrics[b, j] = np.array(r[0])

    subfolder = "upkie_" + datetime.now().strftime("%b_%d_%Y_%H%M") + "/"
    if not os.path.exists(os.path.join(high_level_folder, subfolder)):
        os.makedirs(os.path.join(high_level_folder, subfolder), exist_ok=True)
    with h5py.File(os.path.join(high_level_folder, subfolder, "dataset.hdf5"), "w") as f:
        f.create_dataset("intrinsics", shape=intrinsics.shape)
        f.create_dataset("gains", shape=(num_batches, batch_size, len(MPCBalancerParams().get_list())))
        f.create_dataset("reference_tracks_enc", shape=(num_batches, batch_size, 4))
        f.create_dataset("metrics", shape=(num_batches, batch_size, len(DelayedUpkieSystem.perf_metric_names())))
        f["intrinsics"][...] = intrinsics
        f["gains"][...] = gains
        f["reference_tracks_enc"][...] = ref_tracks_enc
        f["metrics"][...] = metrics
