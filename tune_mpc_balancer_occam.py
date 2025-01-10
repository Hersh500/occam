import numpy as np
import torch
from omegaconf import OmegaConf
import os
import hydra
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import gin
import gymnasium as gym
import numpy as np

# Upkie Imports
import upkie.envs
from qpmpc import MPCQP, Plan
from qpmpc.systems import WheeledInvertedPendulum
from qpsolvers import solve_problem
from upkie.utils.clamp import clamp_and_warn
from upkie.utils.filters import low_pass_filter
from upkie.utils.raspi import configure_agent_process, on_raspi
from upkie.utils.spdlog import logging
from learned_ctrlr_opt.systems.upkie.proxqp_workspace import ProxQPWorkspace

# OCCAM Imports
from learned_ctrlr_opt.meta_learning.occam_model import OCCAMModel
from learned_ctrlr_opt.systems.upkie.upkie import WheeledInvertedPendulumAngularAccMPC

upkie.envs.register()


@gin.configurable
@dataclass
class UpkieConfig:
    leg_length: float
    max_ground_velocity: float
    wheel_radius: float

    rotation_base_to_imu: Optional[List[float]] = None

    def get_spine_config(self) -> dict:
        spine_config = {
            "wheel_odometry": {
                "signed_radius": {
                    "left_wheel": +self.wheel_radius,
                    "right_wheel": -self.wheel_radius,
                }
            }
        }
        if self.rotation_base_to_imu is not None:
            spine_config["base_orientation"] = {
                "rotation_base_to_imu": np.array(
                    self.rotation_base_to_imu,
                    dtype=float,
                ),
            }
        return spine_config

# Runs the balancer MPC in series with OCCAM. Much simpler, but on a low-compute platform,
# OCCAM could cause issues by periodically blocking control computation.
@gin.configurable
def tune_balancer_synchronous(env: gym.Env,
                              max_ground_accel: float,
                              mpc_sampling_period: float,
                              nb_mpc_timesteps: int,
                              stage_input_cost_weight: float,
                              stage_state_cost_weight: float,
                              terminal_cost_weight: float,
                              warm_start: bool,
                              occam_cfg: OmegaConf):

    occam_model = OCCAMModel(occam_cfg,
                             path_header="",
                             task_input=True)

    observation, info = env.reset()  # connects to the spine
    (
        base_pitch,
        ground_position,
        base_angular_velocity,
        ground_velocity,
    ) = observation
    occam_initial_state = np.array(
        [
            0.0,
            base_pitch,
            ground_velocity,
            base_angular_velocity,
        ]
    )

    upkie_config = UpkieConfig()

    # currently gets initial parameters from the config.
    leg_length = occam_cfg.initial_params[0]
    wheel_radius = occam_cfg.initial_params[1]
    best_gain = np.array([leg_length, wheel_radius])

    pendulum = WheeledInvertedPendulum(
        length=leg_length,
        max_ground_accel=max_ground_accel,
        nb_timesteps=nb_mpc_timesteps,
        sampling_period=mpc_sampling_period,
    )
    mpc_problem = pendulum.build_mpc_problem(
        terminal_cost_weight=terminal_cost_weight,
        stage_state_cost_weight=stage_state_cost_weight,
        stage_input_cost_weight=stage_input_cost_weight,
    )
    env.wheel_radius = wheel_radius  # used for converting linear velocity command to angular velocity

    mpc_problem.initial_state = np.zeros(4)
    mpc_qp = MPCQP(mpc_problem)
    workspace = ProxQPWorkspace(mpc_qp)

    commanded_velocity = 0.0
    action = np.zeros(env.action_space.shape)

    # run OCCAM every couple of iterations, then replace gains.
    # fully copied from https://github.com/upkie/mpc_balancer/blob/main/run_agent.py
    t = 0
    states = []
    inputs = []
    while True:
        action[0] = commanded_velocity
        observation, _, terminated, truncated, info = env.step(action)
        env.unwrapped.log("observation", observation)
        if terminated or truncated:
            observation, info = env.reset()
            commanded_velocity = 0.0

        spine_observation = info["spine_observation"]
        floor_contact = spine_observation["floor_contact"]["contact"]

        # Unpack observation into initial MPC state
        (
            base_pitch,
            ground_position,
            base_angular_velocity,
            ground_velocity,
        ) = observation
        current_state = np.array(
            [
                ground_position,
                base_pitch,
                ground_velocity,
                base_angular_velocity,
            ]
        )
        if occam_cfg.tune_with_occam:
            states.append(current_state)

        nx = WheeledInvertedPendulum.STATE_DIM
        target_states = np.zeros((pendulum.nb_timesteps + 1) * nx)
        mpc_problem.update_initial_state(current_state)
        mpc_problem.update_goal_state(target_states[-nx:])
        mpc_problem.update_target_states(target_states[:-nx])

        mpc_qp.update_cost_vector(mpc_problem)
        if warm_start:
            qpsol = workspace.solve(mpc_qp)
        else:
            qpsol = solve_problem(mpc_qp.problem, solver="proxqp")
        if not qpsol.found:
            logging.warning("No solution found to the MPC problem")
        plan = Plan(mpc_problem, qpsol)

        if not floor_contact:
            commanded_velocity = low_pass_filter(
                prev_output=commanded_velocity,
                cutoff_period=0.1,
                new_input=0.0,
                dt=env.unwrapped.dt,
            )
        elif plan.is_empty:
            logging.error("Solver found no solution to the MPC problem")
            logging.info("Continuing with previous action")
        else:  # plan was found
            pendulum.state = current_state
            commanded_accel = plan.first_input[0]
            if occam_cfg.tune_with_occam:
                inputs.append(commanded_accel / wheel_radius)
            commanded_velocity = clamp_and_warn(
                commanded_velocity + commanded_accel * env.unwrapped.dt / 2.0,
                lower=-upkie_config.max_ground_velocity,
                upper=upkie_config.max_ground_velocity,
                label="commanded_velocity",
            )

        t += env.unwrapped.dt

        if occam_cfg.tune_with_occam and t >= occam_cfg.occam_update_freq:
            # running occam...
            states_np = np.array(states)
            inputs_np = np.array(inputs)
            # compute performance measures
            velocity_error = np.sum(np.abs(states_np[:,2])) / states_np.shape[0]
            angle_error = np.sum(np.abs(states_np[:,1])) / states_np.shape[0]
            effort = np.sum(np.abs(inputs_np)) / inputs_np.shape[0]
            performance = np.array([velocity_error, angle_error, effort])

            # adapt model
            occam_model.adapt_model(best_gain, performance, occam_initial_state)

            # search for new gains.
            best_gain, other_info = occam_model.optimize_random_search(task_input=np.array([0.0,
                                                                                            current_state[1],
                                                                                            current_state[2],
                                                                                            current_state[3]]))

            leg_length, wheel_radius = best_gain[0], best_gain[1]
            logging.info(f"setting leg length = {leg_length}")
            logging.info(f"setting wheel radius = {wheel_radius}")
            pendulum.length = leg_length
            env.wheel_radius = wheel_radius
            occam_initial_state = current_state
            occam_initial_state[0] = 0.0

            # Rebuild MPC Problem
            pendulum = WheeledInvertedPendulum(
                length=leg_length,
                max_ground_accel=max_ground_accel,
                nb_timesteps=nb_mpc_timesteps,
                sampling_period=mpc_sampling_period,
            )
            mpc_problem = pendulum.build_mpc_problem(
                terminal_cost_weight=terminal_cost_weight,
                stage_state_cost_weight=stage_state_cost_weight,
                stage_input_cost_weight=stage_input_cost_weight,
            )

            mpc_problem.initial_state = np.zeros(4)
            mpc_qp = MPCQP(mpc_problem)
            workspace = ProxQPWorkspace(mpc_qp)
            t = 0


# Runs the balancer MPC in parallel with OCCAM, requiring two threads. This way, the potentially
# slow random search step does not block the computation of the controller.
def tune_balancer_asynchronous():
    return


def parse_gin_config():
    hostname = socket.gethostname()
    config_dir = Path(__file__).parent / "configs"
    gin.parse_config_file(f"{config_dir}/base.gin")
    host_config = Path(config_dir / f"{hostname}.gin")
    if host_config.exists():
        gin.parse_config_file(host_config)


def main():
    cfg = OmegaConf.load("configs/upkie_tuning.yaml")
    upkie_config = UpkieConfig()
    logging.info(f"Leg length: {upkie_config.leg_length} m")
    logging.info(f"Wheel radius: {upkie_config.wheel_radius} m")
    with gym.make(
        "UpkieGroundVelocity-v3",
        disable_env_checker=True,  # faster startup
        frequency=200.0,
        max_ground_velocity=upkie_config.max_ground_velocity,
        spine_config=upkie_config.get_spine_config(),
        wheel_radius=upkie_config.wheel_radius,
    ) as env:
        tune_balancer_synchronous(env=env, occam_cfg=cfg)


if __name__ == "__main__":
    if on_raspi():
        configure_agent_process()
    parse_gin_config()
    main()
