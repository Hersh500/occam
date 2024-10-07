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

from learned_ctrlr_opt.meta_learning.reptile_net import *
from learned_ctrlr_opt.meta_learning.basis_kf import *
from learned_ctrlr_opt.meta_learning.utils import get_sysid_history_dataset, get_scalers
from learned_ctrlr_opt.opt.optimize_thru_network import *
from learned_ctrlr_opt.opt.random_search import *
from learned_ctrlr_opt.opt.particle_search import *
from learned_ctrlr_opt.systems.car_controller import pp_track_curvature
from learned_ctrlr_opt.systems.car_dynamics import CarParams
from learned_ctrlr_opt.systems.robots import TopDownCarResetFree
from learned_ctrlr_opt.utils.experiment_utils import *
from learned_ctrlr_opt.eval.eval_utils import *

def eval_success(traj):
    return np.abs(traj[-1][2]) < 15

def tdc_rf_reptile_runner(experiment_cfg, random_seed):
    reptile_checkpoint_dir = experiment_cfg.reptile_ckpt_dir
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    reptile_cfg = OmegaConf.load(os.path.join(reptile_checkpoint_dir, "config.yaml"))

    reptile_net, gain_scaler, history_scaler, metric_scaler = load_reptile_and_scalers(experiment_cfg)

    dset_f = h5py.File(reptile_cfg.path_to_dataset, 'r')
    ref_tracks_enc = np.array(dset_f["reference_tracks_enc"])
    ref_track_scaler = MinMaxScaler(clip=True).fit(ref_tracks_enc.reshape(-1, 1))
    dset_f.close()

    gain_dim = len(reptile_cfg.gains_to_optimize)
    lookahead_dim = experiment_cfg.lookahead_dim
    history_in_size = reptile_cfg.history_length * reptile_cfg.traj_dim

    test_set = experiment_cfg.test_set_name
    robot_name = experiment_cfg.robot_name
    params_array, seeds, robot_kwargs = load_test_set(robot_name, test_set)
    params_array = [CarParams(*params_array[i]) for i in range(params_array.shape[0])]

    cost_weights = np.array(experiment_cfg.cost_weights)
    sigma_weight_start = experiment_cfg.kf_sigma_penalty
    initial_gain = np.array(experiment_cfg.initial_gain)
    num_trials = experiment_cfg.num_trials

    failed_count = 0
    expected_costs_reptile = np.zeros((len(seeds), len(params_array), num_trials, 1))
    actual_costs_reptile = np.zeros((len(seeds), len(params_array), num_trials, 1))
    raw_metrics_reptile = np.zeros((len(seeds), len(params_array), num_trials, len(reptile_cfg.metric_idxs)))
    tried_gains_reptile = np.zeros((len(seeds), len(params_array), num_trials, gain_dim))
    expected_scaled_metrics_reptile = np.zeros((len(seeds), len(params_array), num_trials, len(reptile_cfg.metric_idxs)))
    actual_scaled_metrics_reptile = np.zeros((len(seeds), len(params_array), num_trials, len(reptile_cfg.metric_idxs)))
    track_segments = np.zeros(
        (len(seeds), len(params_array), num_trials, int(lookahead_dim * (reptile_cfg.ref_track_ds_factor)), 2))

    try:
        for s_idx, seed in enumerate(seeds):
            for p_idx, params in enumerate(params_array):
                print(f"Seed: {random_seed}: on p_idx {p_idx}")
                robot = TopDownCarResetFree(seed=int(seed),
                                            car_params=params,
                                            gains_to_optimize=reptile_cfg.gains_to_optimize,
                                            length=int(lookahead_dim * reptile_cfg.ref_track_ds_factor),
                                            max_time=robot_kwargs["max_time"])

                metrics, traj, env = robot.evaluate_x(initial_gain, render=False)
                success = eval_success(traj)
                if not success:
                    print("Could not find an initial gain that worked!")
                    break

                task_input_data = torch.zeros(num_trials, gain_dim + lookahead_dim + history_in_size)
                task_target_data = torch.zeros(num_trials, len(reptile_cfg.metric_idxs))
                adapted_net = reptile_net
                np.random.seed(random_seed+p_idx+s_idx+10000)
                torch.manual_seed(random_seed+p_idx+s_idx+10000)
                for i in range(num_trials):
                    def eval_fn(x, cost_weights, sigma_weight):
                        q = x.size(0)
                        cost_weights = torch.from_numpy(cost_weights)
                        ys = adapted_net(x.float())
                        cost_weights_batch = cost_weights.repeat(q).reshape((q, cost_weights.shape[-1])).to(device)
                        mean_losses = torch.sum(ys * cost_weights_batch, dim=-1)
                        losses = mean_losses
                        return losses, ys, torch.zeros(*losses.shape)

                    if not eval_success(traj):
                        failed_count += 1
                        print("went out of bounds!")
                        break

                    traj_lim = traj[-reptile_cfg.history_length:]
                    traj_lim = np.expand_dims(traj_lim, 0)
                    traj_rs = traj_lim.reshape(-1, traj_lim.shape[-1])
                    traj_scaled = history_scaler.transform(traj_rs)
                    traj_flat = np.squeeze(traj_scaled.reshape(traj_lim.shape[0], -1))
                    track_ahead = robot.get_track_from(env)
                    track_segments[s_idx, p_idx, i] = track_ahead
                    track_ahead_pp = pp_track_curvature(track_ahead, ref_track_scaler, reptile_cfg.ref_track_ds_factor)
                    track_ahead_pp_torch = torch.from_numpy(track_ahead_pp).squeeze()

                    if i > 0:
                        perturbed_optima = [task_input_data[:i, :gain_dim]]
                        for num_to_add in range(10):
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
                        guaranteed_searches=both_task_input_data[:i, :gain_dim])

                    tried_gains_reptile[s_idx, p_idx, i, :] = best_gain.detach().cpu().numpy()
                    best_gain_unscaled = np.squeeze(
                        gain_scaler.inverse_transform(best_gain.detach().cpu().numpy().reshape(1, -1)))
                    # print(f"---- Step {i} ----")
                    # print(f"best_gain_unscaled = {best_gain_unscaled}")
                    perf_metrics, traj, env = robot.evaluate_x(best_gain_unscaled, env, render=False)
                    perf_metrics[..., reptile_cfg.metric_idxs_to_invert] = 1 / (
                            1 + perf_metrics[..., reptile_cfg.metric_idxs_to_invert])
                    raw_metrics_reptile[s_idx, p_idx, i] = perf_metrics[reptile_cfg.metric_idxs]
                    # print(f"perf metrics were {perf_metrics}")
                    # print(
                    #     f"scaled perf metrics were {metric_scaler.transform(perf_metrics[reptile_cfg.metric_idxs].reshape(1, -1)).squeeze()}")
                    true_y_scaled = np.dot(cost_weights, metric_scaler.transform(
                        perf_metrics[reptile_cfg.metric_idxs].reshape(1, -1)).squeeze())

                    expected_costs_reptile[s_idx, p_idx, i] = best_y
                    actual_costs_reptile[s_idx, p_idx, i] = true_y_scaled
                    actual_scaled_metrics_reptile[s_idx, p_idx, i] = metric_scaler.transform(
                        perf_metrics[reptile_cfg.metric_idxs].reshape(1, -1)).squeeze()

                    inp = torch.cat([best_gain, track_ahead_pp_torch, torch.from_numpy(traj_flat)], dim=-1)
                    expected_scaled_metrics_reptile[s_idx, p_idx, i] = adapted_net(inp.unsqueeze(0).float().to(device)).cpu().detach().numpy().squeeze()
                    target = torch.from_numpy(
                        metric_scaler.transform(perf_metrics[reptile_cfg.metric_idxs].reshape(1, -1)).squeeze())
                    task_input_data[i] = inp
                    task_target_data[i] = target
                    adapted_net = adapt_reptile(task_input_data[:i+1],
                                                task_target_data[:i+1],
                                                reptile_net,
                                                reptile_cfg.inner_lr,
                                                reptile_cfg.num_inner_steps)
    finally:
        env.close()

    result_arrays = {"expected_costs": expected_costs_reptile,
                     "actual_costs": actual_costs_reptile,
                     "raw_metrics": raw_metrics_reptile,
                     "tried_gains": tried_gains_reptile,
                     "expected_scaled_metrics": expected_scaled_metrics_reptile,
                     "actual_scaled_metrics": actual_scaled_metrics_reptile,
                     "track_segments": track_segments}
    other_data = {"failed_count":failed_count}
    return result_arrays, other_data 
