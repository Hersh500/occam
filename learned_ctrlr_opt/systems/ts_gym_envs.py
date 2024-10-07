from typing import Optional, Union, List
import gym
from gymnasium import spaces
from gym.core import RenderFrame
import numpy as np

from learned_ctrlr_opt.systems.quadrotor_geom import SE3ControlGains, QuadrotorSE3Control_ResetFree, \
    CrazyFlieParamsTrain, generate_random_circle_traj, eval_success
from learned_ctrlr_opt.systems.car_controller import CarControllerParams, pp_track_curvature
from learned_ctrlr_opt.systems.car_dynamics import CarParamsTrain
from learned_ctrlr_opt.systems.robots import TopDownCarResetFree

from learned_ctrlr_opt.utils.dataset_utils import normalize, denormalize

# No tasks in this env, yet.
class CrazyflieSE3ControlGym(gym.Env):
    def __init__(self,
                 history_length,
                 traj_dim,
                 thetas_to_randomize,
                 theta_type,
                 gains_to_optimize,
                 metric_idxs,
                 metrics_to_invert,
                 cost_weights,
                 metric_scaler,
                 history_scaler,
                 step_tf,
                 Tmax):
        super().__init__()
        self.thetas_to_randomize = thetas_to_randomize
        self.theta_t = theta_type
        self.gains_to_optimize = gains_to_optimize
        self.metric_idxs = metric_idxs
        self.metrics_to_invert = metrics_to_invert
        self.cost_weights = cost_weights
        self.metric_scaler = metric_scaler
        self.Tmax = Tmax
        self.step_tf = step_tf
        self.history_length = history_length
        self.traj_dim = traj_dim
        self.history_scaler = history_scaler

        self.num_steps = 0
        self.internal_env = None
        self.env_state = None

        self.action_dim = len(gains_to_optimize)
        self.state_dim = history_length * traj_dim
        self.gain_bounds = SE3ControlGains.get_bounds()[gains_to_optimize]
        self.action_space = spaces.Box(low=np.zeros(len(gains_to_optimize)), high=np.ones(len(gains_to_optimize)))
        # self.observation_space = spaces.Dict({"theta": spaces.Box(low=np.zeros(len(thetas_to_randomize)), high=np.ones(len(thetas_to_randomize))),
        #                                       "state": spaces.Box(low=np.zeros(traj_dim), high=np.ones(traj_dim))})
        self.observation_space = spaces.Box(low=np.zeros(len(thetas_to_randomize)+traj_dim), high=np.ones(len(thetas_to_randomize)+traj_dim))

    def step(self, action):
        if self.internal_env is None:
            raise RuntimeError(f"Must call reset first!")

        g = denormalize(action, self.gain_bounds)
        metrics, actual_traj, results = self.internal_env.evaluate_x(g, raw_pos=False, initial_state=self.env_state)
        metrics[self.metrics_to_invert] = 1/(1+metrics[self.metrics_to_invert])
        metrics = self.metric_scaler.transform(metrics[self.metric_idxs].reshape(1, -1)).squeeze()
        reward = np.dot(self.cost_weights, metrics)
        state = self.history_scaler.transform(actual_traj[-1:]).squeeze()
        theta = normalize(self.internal_env.params.get_list()[self.thetas_to_randomize], self.theta_t.get_bounds())
        self.num_steps += 1
        done = eval_success(results["state"]["x"])
        trunc = self.num_steps > self.Tmax
        info = {"full_traj": actual_traj}
        self.env_state = results
        return np.concatenate((state, theta), axis=0), reward, done, trunc, info


    def reset(self, seed=None, options=None, params=None, traj=None):
        super().reset(seed=seed, options=options)
        if params is None:
            params = self.theta_t.generate_random(self.thetas_to_randomize)
        if traj is None:
            traj_obj, traj_params = generate_random_circle_traj(0.7)

        self.internal_env = QuadrotorSE3Control_ResetFree(params=params,
                                                          gains_to_optimize=self.gains_to_optimize,
                                                          trajectory_obj=traj_obj,
                                                          t_f=self.step_tf)
        random_gains = SE3ControlGains.generate_random(self.gains_to_optimize).get_list()[self.gains_to_optimize]
        metrics, actual_traj, results = self.internal_env.evaluate_x(random_gains, raw_pos=False)
        self.env_state = results
        info = {"full_traj": actual_traj}
        state = self.history_scaler.transform(actual_traj[-1:]).squeeze()
        theta = normalize(self.internal_env.params.get_list()[self.thetas_to_randomize], self.theta_t.get_bounds())
        self.num_steps = 0
        return np.concatenate((state, theta), axis=0), info

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

def tdc_eval_success(traj):
    return np.abs(traj[-1][2]) < 15

class TDCControlGym(gym.Env):
    def __init__(self,
                 history_length,
                 traj_dim,
                 thetas_to_randomize,
                 theta_type,
                 gains_to_optimize,
                 metric_idxs,
                 metrics_to_invert,
                 cost_weights,
                 metric_scaler,
                 history_scaler,
                 step_tf,
                 ref_track_scaler,
                 Tmax,
                 ds_factor=3):
        super().__init__()
        self.thetas_to_randomize = thetas_to_randomize
        self.theta_t = theta_type
        self.gains_to_optimize = gains_to_optimize
        self.metric_idxs = metric_idxs
        self.metrics_to_invert = metrics_to_invert
        self.cost_weights = cost_weights
        self.metric_scaler = metric_scaler
        self.ref_track_scaler = ref_track_scaler
        self.Tmax = Tmax
        self.ds_factor = ds_factor
        self.step_tf = step_tf
        self.history_length = history_length
        self.traj_dim = traj_dim
        self.history_scaler = history_scaler

        self.num_steps = 0
        self.internal_env = None
        self.env_state = None
        self.task_dim = int(step_tf/ds_factor)

        self.action_dim = len(gains_to_optimize)
        self.state_dim = history_length * traj_dim
        self.gain_bounds = SE3ControlGains.get_bounds()[gains_to_optimize]
        self.action_space = spaces.Box(low=np.zeros(len(gains_to_optimize)), high=np.ones(len(gains_to_optimize)))
        self.observation_space = spaces.Box(low=np.zeros(int(step_tf/ds_factor)+len(thetas_to_randomize)+traj_dim),
                                            high=np.ones(int(step_tf/ds_factor)+len(thetas_to_randomize)+traj_dim))

    def step(self, action):
        if self.internal_env is None:
            raise RuntimeError(f"Must call reset first!")

        g = denormalize(action, self.gain_bounds)
        metrics, actual_traj, env_state = self.internal_env.evaluate_x(g, env=self.env_state, render=False)
        metrics[self.metrics_to_invert] = 1/(1+metrics[self.metrics_to_invert])
        metrics = self.metric_scaler.transform(metrics[self.metric_idxs].reshape(1, -1)).squeeze()
        reward = np.dot(self.cost_weights, metrics)
        state = self.history_scaler.transform(actual_traj[-1:]).squeeze()
        theta = normalize(self.internal_env.params.get_list()[self.thetas_to_randomize], self.theta_t.get_bounds()[self.thetas_to_randomize])
        self.num_steps += 1
        done = tdc_eval_success(actual_traj)
        trunc = self.num_steps > self.Tmax
        info = {"full_traj": actual_traj}
        self.env_state = env_state
        track_ahead = self.internal_env.get_track_from(self.env_state)
        track_ahead_pp = pp_track_curvature(track_ahead, self.ref_track_scaler, self.ds_factor).squeeze()
        return np.concatenate((track_ahead_pp, state, theta), axis=0), reward, done, trunc, info

    def reset(self, seed=None, options=None, params=None):
        super().reset(seed=seed, options=options)
        if params is None:
            params = self.theta_t.generate_random(self.thetas_to_randomize)
        if seed is None:
            seed = np.random.randint(0, 10000)
        self.internal_env = TopDownCarResetFree(seed=seed,
                                                car_params=params,
                                                gains_to_optimize=self.gains_to_optimize,
                                                length=self.step_tf,
                                                max_time=1500)
        random_gains = CarControllerParams.generate_random(self.gains_to_optimize).get_list()[self.gains_to_optimize]
        metrics, actual_traj, results = self.internal_env.evaluate_x(random_gains, render=False)
        self.env_state = results
        info = {"full_traj": actual_traj}
        track_ahead = self.internal_env.get_track_from(self.env_state)
        track_ahead_pp = pp_track_curvature(track_ahead, self.ref_track_scaler, self.ds_factor).squeeze()
        state = self.history_scaler.transform(actual_traj[-1:]).squeeze()
        theta = normalize(self.internal_env.params.get_list()[self.thetas_to_randomize], self.theta_t.get_bounds()[self.thetas_to_randomize])
        self.num_steps = 0
        return np.concatenate((track_ahead_pp, state, theta), axis=0), info

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass
