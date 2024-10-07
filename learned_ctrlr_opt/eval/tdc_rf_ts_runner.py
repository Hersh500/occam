import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch.optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader
from learned_ctrlr_opt.utils.learning_utils import init_weights, create_network
from datetime import datetime
import wandb
import os
import hydra
from omegaconf import OmegaConf
import gym

from learned_ctrlr_opt.meta_learning.teacher_student import *
from learned_ctrlr_opt.meta_learning.utils import get_sysid_history_dataset, get_scalers
from learned_ctrlr_opt.opt.random_search import *
from learned_ctrlr_opt.opt.particle_search import *
from learned_ctrlr_opt.systems.car_controller import pp_track_curvature
from learned_ctrlr_opt.systems.car_dynamics import CarParams
from learned_ctrlr_opt.systems.robots import TopDownCarResetFree
from learned_ctrlr_opt.utils.experiment_utils import *
from learned_ctrlr_opt.eval.eval_utils import *


def eval_success(traj):
    return np.abs(traj[-1][2]) < 15


def tdc_rf_ts_runner(experiment_cfg, random_seed):
    ts_checkpoint_dir = experiment_cfg.ts_ckpt_dir
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ts_cfg = OmegaConf.load(os.path.join(ts_checkpoint_dir, "config.yaml"))
    gain_dim = len(ts_cfg.gains_to_optimize)
    lookahead_dim = experiment_cfg.lookahead_dim
    history_in_size = ts_cfg.history_length * ts_cfg.traj_dim

    ts_network, gain_scaler, history_scaler, metric_scaler = load_ts_and_scalers(experiment_cfg)
    _, _, _, metric_scaler = load_kf_and_scalers(experiment_cfg)  # USE KF SCALER FOR CONSISTENCY AMONGST DATASETS

    dset_f = h5py.File(ts_cfg.path_to_dataset, 'r')
    ref_tracks_enc = np.array(dset_f["reference_tracks_enc"])
    ref_track_scaler = MinMaxScaler(clip=True).fit(ref_tracks_enc.reshape(-1, 1))
    dset_f.close()

    test_set = experiment_cfg.test_set_name
    robot_name = experiment_cfg.robot_name
    params_array, seeds, robot_kwargs = load_test_set(robot_name, test_set)
    params_array = [CarParams(*params_array[i]) for i in range(params_array.shape[0])]

    cost_weights = np.array(experiment_cfg.cost_weights)
    initial_gain = np.array(experiment_cfg.initial_gain)
    num_trials = experiment_cfg.num_trials

    expected_costs = np.zeros((len(seeds), len(params_array), num_trials, 1))
    actual_costs = np.zeros((len(seeds), len(params_array), num_trials, 1))
    raw_metrics = np.zeros((len(seeds), len(params_array), num_trials, len(ts_cfg.metric_idxs)))
    tried_gains = np.zeros((len(seeds), len(params_array), num_trials, len(ts_cfg.gains_to_optimize)))
    expected_scaled_metrics = np.zeros((len(seeds), len(params_array), num_trials, len(ts_cfg.metric_idxs)))
    actual_scaled_metrics = np.zeros((len(seeds), len(params_array), num_trials, len(ts_cfg.metric_idxs)))
    latents = np.zeros((len(seeds), len(params_array), num_trials, ts_cfg.latent_dim))
    track_segments = np.zeros(
        (len(seeds), len(params_array), num_trials, int(experiment_cfg.lookahead_dim * ts_cfg.ref_track_ds_factor), 2))
    failed_count = 0
    try:
        for s_idx, seed in enumerate(seeds):
            for p_idx, params in enumerate(params_array):
                print(f"Seed: {random_seed}: on p_idx {p_idx}")
                robot = TopDownCarResetFree(seed=int(seed),
                                            car_params=params,
                                            gains_to_optimize=ts_cfg.gains_to_optimize,
                                            length=experiment_cfg.segment_length,
                                            max_time=robot_kwargs["max_time"])

                success = False
                metrics, traj, env = robot.evaluate_x(initial_gain, render=False)
                success = eval_success(traj)
                if not success:
                    print("Could not find an initial gain that worked!")
                    break

                task_input_data = torch.zeros(num_trials, gain_dim + lookahead_dim + history_in_size)
                task_target_data = torch.zeros(num_trials, len(ts_cfg.metric_idxs))
                np.random.seed(random_seed+p_idx+s_idx)
                torch.manual_seed(random_seed+p_idx+s_idx)
                for i in range(num_trials):
                    def eval_fn(x, cost_weights, sigma_weight):
                        q = x.size(0)
                        cost_weights = torch.from_numpy(cost_weights).float().to(device)
                        ys = ts_network(x.float().to(device))
                        cost_weights_batch = cost_weights.repeat(q).reshape((q, cost_weights.shape[-1])).to(device)
                        mean_losses = torch.sum(ys * cost_weights_batch, dim=-1)
                        variances = torch.zeros(x.size(0), device=device)
                        return mean_losses, ys, variances

                    if not eval_success(traj):
                        print("went out of bounds!")
                        failed_count += 1
                        break

                    traj_lim = traj[-ts_cfg.history_length:]
                    traj_lim = np.expand_dims(traj_lim, 0)
                    traj_rs = traj_lim.reshape(-1, traj_lim.shape[-1])
                    traj_scaled = history_scaler.transform(traj_rs)
                    traj_flat = np.squeeze(traj_scaled.reshape(traj_lim.shape[0], -1))
                    track_ahead = robot.get_track_from(env)  # need some way to paramtrize this from the config...
                    track_segments[s_idx, p_idx, i] = track_ahead

                    track_ahead_pp = pp_track_curvature(track_ahead, ref_track_scaler, ts_cfg.ref_track_ds_factor)
                    track_ahead_pp_torch = torch.from_numpy(track_ahead_pp).squeeze()
                    if i > 0:
                        perturbed_optima = [task_input_data[:i, :gain_dim]]
                        for num_to_add in range(5):
                            noisy_task_input_data = torch.clip(task_input_data[:i, :gain_dim] + torch.randn(
                                task_input_data[:i, :gain_dim].shape) * 3e-2, 0, 1)
                            perturbed_optima.append(noisy_task_input_data)
                        both_task_input_data = torch.cat(perturbed_optima, dim=0)
                    else:
                        both_task_input_data = task_input_data[:i, :gain_dim]

                    best_gain, best_cost, best_y, all_tested_gains, all_tested_ys, all_tested_costs = random_search(
                        eval_fn,
                        experiment_cfg.num_search_samples,
                        cost_weights,
                        gain_dim,
                        device,
                        torch.cat([track_ahead_pp_torch, torch.from_numpy(traj_flat)]),
                        batch_size=512,
                        sigma_weight=0,
                        # guaranteed_searches=None)
                        guaranteed_searches=both_task_input_data)

                    # should I save unscaled or scaled gains?
                    tried_gains[s_idx, p_idx, i, :] = best_gain.detach().cpu().numpy()
                    best_gain_unscaled = np.squeeze(
                        gain_scaler.inverse_transform(best_gain.detach().cpu().numpy().reshape(1, -1)))
                    # print(f"---- Step {i} ----")
                    # print(f"best_gain_unscaled = {best_gain_unscaled}")
                    perf_metrics, traj, env = robot.evaluate_x(best_gain_unscaled, env, render=False)
                    perf_metrics[..., ts_cfg.metric_idxs_to_invert] = 1 / (
                                1 + perf_metrics[..., ts_cfg.metric_idxs_to_invert])
                    raw_metrics[s_idx, p_idx, i] = perf_metrics[ts_cfg.metric_idxs]
                    # print(f"perf metrics were {perf_metrics}")
                    # print(
                    #     f"scaled perf metrics were {metric_scaler.transform(perf_metrics[kf_cfg.metric_idxs].reshape(1, -1)).squeeze()}")
                    true_y_scaled = np.dot(cost_weights, metric_scaler.transform(
                        perf_metrics[ts_cfg.metric_idxs].reshape(1, -1)).squeeze())

                    expected_costs[s_idx, p_idx, i] = best_y
                    actual_costs[s_idx, p_idx, i] = true_y_scaled
                    actual_scaled_metrics[s_idx, p_idx, i] = metric_scaler.transform(
                        perf_metrics[ts_cfg.metric_idxs].reshape(1, -1)).squeeze()

                    input_test = torch.cat([best_gain, track_ahead_pp_torch, torch.from_numpy(traj_flat)],
                                           dim=-1).unsqueeze(0)
                    best_y_mean = ts_network(input_test.float().to(device))
                    expected_scaled_metrics[s_idx, p_idx, i] = best_y_mean.cpu().detach().numpy()

                    inp = torch.cat([best_gain, track_ahead_pp_torch, torch.from_numpy(traj_flat)], dim=-1)
                    target = torch.from_numpy(
                        metric_scaler.transform(perf_metrics[ts_cfg.metric_idxs].reshape(1, -1)).squeeze())
                    task_input_data[i] = inp
                    task_target_data[i] = target
                    latent = ts_network.encode_student(torch.from_numpy(traj_flat).unsqueeze(0).float().to(device))
                    latents[s_idx, p_idx, i] = latent.cpu().detach().numpy()
    finally:
        env.close()

    result_arrays = {"expected_costs": expected_costs,
                     "actual_costs": actual_costs,
                     "raw_metrics": raw_metrics,
                     "tried_gains": tried_gains,
                     "latents": latents,
                     "expected_scaled_metrics": expected_scaled_metrics,
                     "actual_scaled_metrics": actual_scaled_metrics,
                     "track_segments": track_segments}
    other_data = {"failed_count":failed_count}
    return result_arrays, other_data
