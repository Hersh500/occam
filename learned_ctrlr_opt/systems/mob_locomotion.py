import gc

import numpy as np
from dataclasses import dataclass
import glob
import pickle as pkl

try:
    from go1_gym.envs.base.legged_robot_config import Cfg
    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
except ModuleNotFoundError or ImportError:
    print("Go1 gym not installed, will not be able to use Mob Loco")

import torch
from typing import List, Union, Optional
from datetime import datetime
import os
import h5py

from learned_ctrlr_opt.systems.robots import Robot
from learned_ctrlr_opt.utils.dataset_utils import denormalize

# NOTE: the code for this system depends on some local modifications to the walk-these-ways codebase
# TODO(hersh500): push those changes to make this code fully usable by others

CMD_BOUNDS = np.array([[-3, 3],
                       [-1, 1],
                       [-3, 3]])

# I don't quite understand how these numbers specify the gaits, but
# this is what's in the paper
GAITS = np.array([[0.0, 0, 0],  # pronking
                  [0.5, 0, 0],  # trotting
                  [0, 0.5, 0],  # bounding,
                  [0, 0, 0.5],  # pacing
                  [0.25, 0, 0]])   # galloping

GAITS_NORM = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [0.5, 0, 0]])
@dataclass
class Go1HiddenParams:
    added_mass: float = 0  # difference between nominal mass and true mass
    motor_strength: float = 1.0
    friction: float = 1.0
    restitution: float = 0.2
    terrain_noise_mag: float = 0.0  # Unused ATM

    def get_list(self):
        return np.array([self.added_mass,
                         self.motor_strength,
                         self.friction,
                         self.restitution,
                         self.terrain_noise_mag])

    @staticmethod
    def get_bounds():
        return np.array([[-1.0, 4.0],
                         [0.8, 1.1],
                         [-0.6, 3.0],   # negative because Isaac Gym averages this value with 1.0
                         [0.05, 0.5],
                         [0.0, 0.0]])

    @staticmethod
    def generate_random(gains_to_randomize):
        random_params = denormalize(np.random.rand(5), Go1HiddenParams.get_bounds())
        params = Go1HiddenParams().get_list()
        params[gains_to_randomize] = random_params[gains_to_randomize]
        return Go1HiddenParams(*list(params))

    @staticmethod
    def get_num():
        return 5

@dataclass
class Go1HiddenParamsTrain(Go1HiddenParams):
    @staticmethod
    def get_bounds():
        return np.array([[-0.8, 2.5],
                         [0.9, 1.0],
                         [-0.5, 2.5],
                         [0.1, 0.3],
                         [0.0, 0.0]])

    @staticmethod
    def generate_random(gains_to_randomize):
        random_params = denormalize(np.random.rand(5), Go1HiddenParamsTrain.get_bounds())
        params = Go1HiddenParamsTrain().get_list()
        params[gains_to_randomize] = random_params[gains_to_randomize]
        return Go1HiddenParamsTrain(*list(params))


@dataclass
class Go1HiddenParamsTest(Go1HiddenParams):
    @staticmethod
    def lower_half_bounds():
        return np.array([[Go1HiddenParams.get_bounds()[i, 0], Go1HiddenParamsTrain.get_bounds()[i, 0]] for i in range(5)])

    @staticmethod
    def upper_half_bounds():
        return np.array([[Go1HiddenParamsTrain.get_bounds()[i, 1], Go1HiddenParams.get_bounds()[i, 1]] for i in range(5)])

    @staticmethod
    def generate_random(to_randomize=None, half=False):
        if to_randomize is None:
            to_randomize = [i for i in range(Go1HiddenParamsTest.get_num())]
        params = Go1HiddenParamsTest().get_list()
        if half:
            lower = np.random.randint(2) * np.ones(len(to_randomize))
        else:
            lower = np.random.randint(2, size=len(to_randomize))
        for idx in to_randomize:
            if lower[idx] == 0:
                bounds = Go1HiddenParamsTest.lower_half_bounds()[idx]
            else:
                bounds = Go1HiddenParamsTest.upper_half_bounds()[idx]
            params[idx] = np.random.uniform(bounds[0], bounds[1])
        return Go1HiddenParamsTest(*params)


@dataclass
class Go1BehaviorParams:
    body_height: float = -0.05
    step_freq: float = 3.0
    gait_phase: float = 0.5
    gait_offset: float = 0.0
    gait_bound: float = 0.0
    gait_duration: float = 0.5
    swing_height: float = 0.19
    body_pitch: float = 0.0
    body_roll: float = 0.0
    stance_width: float = 0.275

    def get_list(self):
        return np.array([self.body_height,
                         self.step_freq,
                         self.gait_phase,
                         self.gait_offset,
                         self.gait_bound,
                         self.gait_duration,
                         self.swing_height,
                         self.body_pitch,
                         self.body_roll,
                         self.stance_width])

    @staticmethod
    def get_bounds():
        return np.array([[-0.15, 0.15],
                         [2.0, 4.0],
                         [0.0, 0.5],
                         [0.0, 0.5],
                         [0.0, 0.5],
                         [0.5, 0.5],
                         [0.07, 0.35],
                         [-0.4, 0.4],
                         [0.0, 0.0],
                         [0.15, 0.45]])

    @staticmethod
    def generate_random(gains_to_randomize):
        random_params = denormalize(np.random.rand(10), Go1BehaviorParams.get_bounds())
        params = Go1BehaviorParams().get_list()
        params[gains_to_randomize] = random_params[gains_to_randomize]
        if 2 in gains_to_randomize and 3 in gains_to_randomize and 4 in gains_to_randomize:
            gait_params_idx = np.random.randint(0, GAITS.shape[0])
            params[[2, 3, 4]] = GAITS[gait_params_idx]
        else:
            params[[2, 3, 4]] = np.array([0.5, 0, 0])  # trot by default
        return Go1BehaviorParams(*params)


# Utilty functions copied from MoB codebase
def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(label, num_envs, headless=False, fixed_params=None, rand_ranges:Optional[Go1HiddenParams]=None):
    dirs = glob.glob(f"/home/hersh/Programming/walk-these-ways/runs/{label}/*")
    logdir = sorted(dirs)[0]

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        cfg = pkl_cfg["Cfg"]
        new_Cfg = Cfg()

        for key, value in cfg.items():
            if hasattr(new_Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(new_Cfg, key), key2, value2)

    if fixed_params is not None:
        new_Cfg.domain_rand.randomize_friction = False
        new_Cfg.domain_rand.randomize_restitution = False
        new_Cfg.domain_rand.randomize_base_mass = False
        new_Cfg.domain_rand.randomize_motor_strength = False
    else:
        assert rand_ranges is not None
        new_Cfg.domain_rand.randomize_friction = True
        new_Cfg.domain_rand.friction_range = rand_ranges.get_bounds()[2]
        new_Cfg.domain_rand.randomize_restitution = True
        new_Cfg.domain_rand.restitution_range= rand_ranges.get_bounds()[3]
        new_Cfg.domain_rand.randomize_base_mass = True
        new_Cfg.domain_rand.added_mass_range = rand_ranges.get_bounds()[3]
        new_Cfg.domain_rand.randomize_motor_strength = True
        new_Cfg.domain_rand.motor_strength_range = rand_ranges.get_bounds()[1]

    new_Cfg.domain_rand.push_robots = False
    new_Cfg.domain_rand.randomize_gravity = False
    new_Cfg.domain_rand.randomize_motor_offset = False
    new_Cfg.domain_rand.randomize_friction_indep = False
    new_Cfg.domain_rand.randomize_ground_friction = False
    new_Cfg.domain_rand.randomize_Kd_factor = False
    new_Cfg.domain_rand.randomize_Kp_factor = False
    new_Cfg.domain_rand.randomize_joint_friction = False
    new_Cfg.domain_rand.randomize_com_displacement = False
    new_Cfg.domain_rand.randomize_rigids_after_start = False

    new_Cfg.env.num_recording_envs = 0
    new_Cfg.env.episode_length_s = 3000  # set to a high value to avoid early resets
    new_Cfg.env.num_envs = num_envs
    new_Cfg.terrain.num_rows = 5
    new_Cfg.terrain.num_cols = 5
    new_Cfg.terrain.border_size = 0
    new_Cfg.terrain.center_robots = True
    new_Cfg.terrain.center_span = 1
    new_Cfg.terrain.teleport_robots = True
    new_Cfg.domain_rand.tile_height_range = [-0.2, 0.2]
    new_Cfg.terrain.terrain_length = 2
    new_Cfg.terrain.terrain_width = 2

    new_Cfg.domain_rand.lag_timesteps = 6
    new_Cfg.domain_rand.randomize_lag_timesteps = True
    new_Cfg.control.control_type = "actuator_net"

    new_Cfg.viewer.pos = [-3, -3, 5]
    new_Cfg.viewer.lookat = [0, 0, 4]

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=new_Cfg, fixed_params=fixed_params)
    env = HistoryWrapper(env)

    policy = load_policy(logdir)
    return env, policy


class MoBLocomotion_ResetFree(Robot):
    ParamsT = Go1HiddenParams
    ControllerParamsT = Go1BehaviorParams
    def __init__(self,
                 gains_to_optimize,
                 params=None,
                 bounds_t=None,
                 num_envs: int = 1,
                 render: bool = False,
                 eval_length_s=1):
        self.gains_to_optimize = gains_to_optimize
        label = "gait-conditioned-agility/pretrain-v0/train"
        if params is not None:
            if isinstance(params, Go1HiddenParams):
                params = params.get_list().reshape(1, -1)
            assert num_envs == params.shape[0]
        self.params = params
        self.num_envs = num_envs
        if params is not None:
            # Might have a cuda issue here.
            params_cuda = torch.from_numpy(self.params).float().cuda()
            params_cuda.requires_grad = False
            self.env, self.policy = load_env(label, num_envs, not render, fixed_params=params_cuda)
        else:
            print(f"Using random parameter ranges!!")
            self.env, self.policy = load_env(label, num_envs, not render, rand_ranges=bounds_t)
        self.obs = self.env.reset()
        self.eval_length_s = eval_length_s
        self.num_steps = int(self.eval_length_s/(self.env.cfg.sim.dt * self.env.cfg.control.decimation))
        self.bounds_t = bounds_t

    def evaluate_x(self,
                   x: Union[List, np.ndarray],
                   cmd: Union[List, np.ndarray],
                   debug=False):

        if self.num_envs == 1 and len(x) != self.num_envs:
            x = np.array([x])
        if self.num_envs == 1 and len(cmd) != self.num_envs:
            cmd = np.array([cmd])

        assert len(x) == self.num_envs
        assert len(cmd) == self.num_envs

        gains = np.zeros((self.num_envs, 10))
        gains[:,:] = Go1BehaviorParams().get_list()
        gains[:,self.gains_to_optimize] = np.array(x)
        gains_torch = torch.from_numpy(gains).cuda()
        cmd_torch = torch.from_numpy(np.array(cmd)).cuda()
        gains_torch.requires_grad = False
        cmd_torch.requires_grad = False

        measured_vels = np.zeros((self.num_envs, self.num_steps, 3))
        # estimated_vels = np.zeros((self.num_envs, num_steps, 3))
        # estimated_torques = np.zeros((self.num_envs, num_steps, 12))
        all_torques = np.zeros((self.num_envs, self.num_steps, self.env.torques.shape[-1]))
        # Torques, body velocities, body angular velocities
        sensor_traj = np.zeros((self.num_envs, self.num_steps, 12+3+3))
        debug_traj = np.zeros((self.num_envs, self.num_steps, 3))  # give robot position as well
        success_idxs = np.ones(self.num_envs, dtype=np.bool)
        for t in range(self.num_steps):
            self.env.commands[:, 0] = cmd_torch[:,0]
            self.env.commands[:, 1] = cmd_torch[:,1]
            self.env.commands[:, 2] = cmd_torch[:,2]
            self.env.commands[:, 3] = gains_torch[:,0]
            self.env.commands[:, 4] = gains_torch[:,1]
            self.env.commands[:, 5:8] = gains_torch[:,2:5]
            self.env.commands[:, 8] = gains_torch[:,5]
            self.env.commands[:, 9] = gains_torch[:,6]
            self.env.commands[:, 10] = gains_torch[:,7]
            self.env.commands[:, 11] = gains_torch[:,8]
            self.env.commands[:, 12] = gains_torch[:,9]
            with torch.no_grad():
                actions = self.policy(self.obs)
                self.obs, rew, done, info = self.env.step(actions)
            success_idxs = np.logical_and(success_idxs,
                                         torch.logical_not(done.flatten()).cpu().detach().numpy())
            measured_vels[success_idxs, t, 0:2] = self.env.base_lin_vel[success_idxs,0:2].detach().cpu().numpy()
            measured_vels[success_idxs, t, 2] = self.env.base_ang_vel[success_idxs,2].detach().cpu().numpy()
            all_torques[success_idxs,t] = self.env.torques[success_idxs,...].detach().cpu().numpy()
            sensor_traj[success_idxs,t,0:3] = self.env.base_lin_vel[success_idxs,...].detach().cpu().numpy()
            sensor_traj[success_idxs,t,3:6] = self.env.base_ang_vel[success_idxs,...].detach().cpu().numpy()
            sensor_traj[success_idxs,t,6:] = self.env.torques[success_idxs,...].detach().cpu().numpy()
            debug_traj[success_idxs,t] = self.env.base_pos[success_idxs,...].detach().cpu().numpy()
        mean_vel_errs = np.mean(np.abs(measured_vels - cmd.reshape((self.num_envs, 1, 3)))**2, axis=1)
        mean_torques = np.mean(np.linalg.norm(all_torques, axis=-1), axis=1)
        if debug:
            return np.hstack([mean_vel_errs, np.expand_dims(mean_torques, 1)]), sensor_traj, success_idxs, debug_traj
        else:
            return np.hstack([mean_vel_errs, np.expand_dims(mean_torques, 1)]), sensor_traj, success_idxs

    def reset_env(self):
        self.obs = self.env.reset()

    @staticmethod
    def perf_metric_names():
        return ["X vel error", "Y vel error", "Yaw vel error", "Motor Torques"]

    def get_gain_bounds(self):
        return self.ControllerParamsT.get_bounds()[self.gains_to_optimize]

    def get_theta_bounds(self):
        if self.bounds_t is not None:
            return self.bounds_t.get_bounds()
        return self.ParamsT.get_bounds()

    def get_nominal_values(self):
        return self.ControllerParamsT().get_list()[self.gains_to_optimize]

    def get_thetas(self):
        masses = self.env.payloads.detach().cpu().numpy().reshape((self.num_envs, 1))
        motor_strengths = self.env.motor_strengths[:,0].detach().cpu().numpy().reshape((self.num_envs, 1))
        frictions = self.env.friction_coeffs[:,0].detach().cpu().numpy().reshape((self.num_envs, 1))
        restitutions = self.env.restitutions[:,0].detach().cpu().numpy().reshape(self.num_envs, 1)
        return np.hstack([masses, motor_strengths, frictions, restitutions])


### Define the parallel processes for gathering data, etc., in here
def gather_batched_mob_loco_sysid_data(n_datapoints,
                                       batch_size,
                                       num_batches_per_intrinsic,
                                       gains_to_randomize,
                                       thetas_to_randomize,
                                       high_level_folder,
                                       initial_sensor_traj_length_tsteps,
                                       randomize_cmds=True,
                                       cmd_bounds=CMD_BOUNDS,
                                       fixed_cmd=[1, 0, 0],
                                       t_f=2,
                                       params_t=Go1HiddenParams):

    assert (randomize_cmds or (not randomize_cmds and fixed_cmd is not None))
    assert n_datapoints == int(n_datapoints / (batch_size * num_batches_per_intrinsic)) * (
            batch_size * num_batches_per_intrinsic)

    num_batches = int(n_datapoints / (batch_size * num_batches_per_intrinsic))
    intrinsics = np.zeros((num_batches, len(thetas_to_randomize)))
    gains_to_test = np.zeros((num_batches, batch_size, len(gains_to_randomize)))

    subfolder = "mob_loco_resetfree_data" + datetime.now().strftime("%b_%d_%Y_%H%M") + "/"
    if not os.path.exists(os.path.join(high_level_folder, subfolder)):
        os.makedirs(os.path.join(high_level_folder, subfolder), exist_ok=True)

    with h5py.File(os.path.join(high_level_folder, subfolder, "dataset.hdf5"), "w") as f:
        f.create_dataset("intrinsics", shape=intrinsics.shape)
        f["intrinsics"].attrs["bounds"] = MoBLocomotion_ResetFree.ParamsT.get_bounds()[thetas_to_randomize]
        f["intrinsics"].attrs["idxs"] = thetas_to_randomize
        f.create_dataset("gains", shape=gains_to_test.shape)
        f.create_dataset("initial_gains", shape=gains_to_test.shape)
        f["gains"].attrs["bounds"] = MoBLocomotion_ResetFree.ControllerParamsT.get_bounds()[gains_to_randomize]
        f["gains"].attrs["idxs"] = gains_to_randomize
        f.create_dataset("trajectories", shape=(num_batches, batch_size, t_f*50, 18))
        f.create_dataset("initial_trajectories", shape=(num_batches, batch_size, initial_sensor_traj_length_tsteps, 18))
        f["initial_trajectories"].attrs["initial_length_tsteps"] = initial_sensor_traj_length_tsteps
        f.create_dataset("metrics", shape=(num_batches, batch_size, len(MoBLocomotion_ResetFree.perf_metric_names())))
        if not randomize_cmds:
            f.create_dataset("reference_tracks", shape=(3,))
            f["reference_tracks"][...] = np.array(fixed_cmd)
        else:
            f.create_dataset("reference_tracks", shape=(num_batches, batch_size, 3))
            f.create_dataset("reference_tracks_enc", shape=(num_batches, batch_size, 3))
        robot = MoBLocomotion_ResetFree(gains_to_randomize,
                                        params=None,
                                        bounds_t=params_t,
                                        num_envs=num_batches,
                                        render=False,
                                        eval_length_s=t_f)
        f["intrinsics"][...] = robot.get_thetas()
        done_counts = np.zeros(num_batches, dtype=int)
        done_check = np.all(done_counts >= batch_size)
        while not done_check:
            robot.reset_env()
            g = np.zeros((num_batches, len(gains_to_randomize)))
            initial_g = np.zeros((num_batches, len(gains_to_randomize)))
            for g_idx in range(num_batches):
                g[g_idx] = Go1BehaviorParams.generate_random(gains_to_randomize).get_list()[gains_to_randomize]
                initial_g[g_idx] = Go1BehaviorParams.generate_random(gains_to_randomize).get_list()[gains_to_randomize]
            if randomize_cmds:
                cmd = denormalize(np.random.rand(num_batches, 3), cmd_bounds)
            else:
                cmd = np.zeros((num_batches, 3))
                cmd[:,0:3] = np.array(fixed_cmd)
            initial_perf_metrics, initial_traj, ig_success_idxs = robot.evaluate_x(initial_g, cmd)
            perf_metrics, traj, successes = robot.evaluate_x(g, cmd)
            for j in range(num_batches):
                if ig_success_idxs[j] and done_counts[j] < batch_size:
                    f["gains"][j, done_counts[j],...] = g[j]
                    f["initial_gains"][j, done_counts[j],...] = initial_g[j]
                    f["metrics"][j, done_counts[j], ...] = perf_metrics[j]
                    f["trajectories"][j, done_counts[j], ...] = traj[j]
                    f["initial_trajectories"][j, done_counts[j], ...] = initial_traj[j][-initial_sensor_traj_length_tsteps:]
                    f["reference_tracks"][j, done_counts[j],...] = cmd[j]
                    f["reference_tracks_enc"][j, done_counts[j],...] = cmd[j]
                    done_counts[j] += 1
            print(f"Finished step {np.max(done_counts[j])}")
            done_check = np.all(done_counts >= batch_size)


def main():
    robot = MoBLocomotion_ResetFree(Go1HiddenParams(),
                                    [i for i in range(10)],
                                    render=True)
    results = robot.evaluate_x(Go1BehaviorParams().get_list(),
                               [1, 0, 0])
    print(results)


if __name__ == "__main__":
    main()
