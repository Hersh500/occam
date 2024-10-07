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

from learned_ctrlr_opt.meta_learning.lsr_net import *
from learned_ctrlr_opt.meta_learning.basis_kf import *
from learned_ctrlr_opt.meta_learning.utils import get_sysid_history_dataset, get_scalers
from learned_ctrlr_opt.meta_learning.lkbo import train_dk_gp
from learned_ctrlr_opt.opt.random_search import *
from learned_ctrlr_opt.opt.particle_search import *
from learned_ctrlr_opt.systems.car_controller import pp_track_curvature
from learned_ctrlr_opt.systems.car_dynamics import CarParams
from learned_ctrlr_opt.systems.robots import TopDownCarResetFree
from learned_ctrlr_opt.utils.experiment_utils import *
from learned_ctrlr_opt.eval.eval_utils import load_kf_and_scalers


def eval_success(traj):
    return np.abs(traj[-1][2]) < 15


def tdc_rf_lkbo_runner(experiment_cfg, random_seed):
    kf_checkpoint_dir = experiment_cfg.kf_ckpt_dir
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    kf_cfg = OmegaConf.load(os.path.join(kf_checkpoint_dir, "config.yaml"))
    gain_dim = len(kf_cfg.gains_to_optimize)
    lookahead_dim = experiment_cfg.lookahead_dim
    history_in_size = kf_cfg.history_length * kf_cfg.traj_dim

    kf_network, gain_scaler, history_scaler, metric_scaler = load_kf_and_scalers(experiment_cfg)

    dset_f = h5py.File(kf_cfg.path_to_dataset, 'r')
    ref_tracks_enc = np.array(dset_f["reference_tracks_enc"])
    ref_track_scaler = MinMaxScaler(clip=True).fit(ref_tracks_enc.reshape(-1, 1))
    dset_f.close()

    test_set = experiment_cfg.test_set_name
    robot_name = experiment_cfg.robot_name
    params_array, seeds, robot_kwargs = load_test_set(robot_name, test_set)
    params_array = [CarParams(*params_array[i]) for i in range(params_array.shape[0])]

    cost_weights = np.array(experiment_cfg.cost_weights)
    sigma_weight_start = experiment_cfg.bo_sigma_penalty
    initial_gain = np.array(experiment_cfg.initial_gain)
    num_trials = experiment_cfg.num_trials

    expected_costs = np.zeros((len(seeds), len(params_array), num_trials, 1))
    actual_costs = np.zeros((len(seeds), len(params_array), num_trials, 1))
    raw_metrics = np.zeros((len(seeds), len(params_array), num_trials, len(kf_cfg.metric_idxs)))
    tried_gains = np.zeros((len(seeds), len(params_array), num_trials, len(kf_cfg.gains_to_optimize)))
    actual_scaled_metrics = np.zeros((len(seeds), len(params_array), num_trials, len(kf_cfg.metric_idxs)))
    track_segments = np.zeros(
        (len(seeds), len(params_array), num_trials, int(experiment_cfg.lookahead_dim * (kf_cfg.ref_track_ds_factor)), 2))
    failed_count = 0
    try:
        for s_idx, seed in enumerate(seeds):
            for p_idx, params in enumerate(params_array):
                print(f"Seed: {random_seed}: on p_idx {p_idx}")
                robot = TopDownCarResetFree(seed=int(seed),
                                            car_params=params,
                                            gains_to_optimize=kf_cfg.gains_to_optimize,
                                            length=experiment_cfg.segment_length,
                                            max_time=robot_kwargs["max_time"])

                success = False
                metrics, traj, env = robot.evaluate_x(initial_gain, render=False)
                success = eval_success(traj)
                if not success:
                    print("Could not find an initial gain that worked!")
                    break

                kf_network.use_last_layer = True

                task_input_data = torch.zeros(num_trials, gain_dim + lookahead_dim + history_in_size)
                task_target_data = torch.zeros(num_trials, 1)
                sigma_weight = sigma_weight_start
                np.random.seed(random_seed+p_idx+s_idx)
                torch.manual_seed(random_seed+p_idx+s_idx)
                for i in range(num_trials):
                    def eval_fn_kf(x, cost_weights, sigma_weight):
                        q = x.size(0)
                        cost_weights = torch.from_numpy(cost_weights).float().to(device)
                        ys = kf_network(x.float().to(device))
                        cost_weights_batch = cost_weights.repeat(q).reshape((q, cost_weights.shape[-1])).to(device)
                        mean_losses = torch.sum(ys * cost_weights_batch, dim=-1)
                        variances = torch.zeros(x.size(0)).to(device)
                        return mean_losses, ys, variances

                    def eval_fn_gp(x, cost_weights, sigma_weight):
                        with torch.no_grad():
                            distr = gp_model(x.float().to(device))
                        mean = distr.mean.detach()
                        variance = distr.variance.detach()
                        loss = mean + sigma_weight*variance
                        return loss, mean.reshape(-1, 1), variance

                    if not eval_success(traj):
                        print("went out of bounds!")
                        failed_count += 1
                        break

                    traj_lim = traj[-kf_cfg.history_length:]
                    traj_lim = np.expand_dims(traj_lim, 0)
                    traj_rs = traj_lim.reshape(-1, traj_lim.shape[-1])
                    traj_scaled = history_scaler.transform(traj_rs)
                    traj_flat = np.squeeze(traj_scaled.reshape(traj_lim.shape[0], -1))
                    track_ahead = robot.get_track_from(env)  # need some way to paramtrize this from the config...
                    track_segments[s_idx, p_idx, i] = track_ahead

                    track_ahead_pp = pp_track_curvature(track_ahead, ref_track_scaler, kf_cfg.ref_track_ds_factor)
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

                    if i <= experiment_cfg.lkbo_num_explore_iters:
                        eval_fn = eval_fn_kf
                        cw = cost_weights
                    else:
                        eval_fn = eval_fn_gp
                        cw = np.array([1])  # GP outputs the scalarized costs already

                    best_gain, best_cost, best_y, all_tested_gains, all_tested_ys, all_tested_costs = random_search(
                        eval_fn,
                        experiment_cfg.num_search_samples,
                        cw,
                        gain_dim,
                        device,
                        torch.cat([track_ahead_pp_torch, torch.from_numpy(traj_flat)]),
                        batch_size=experiment_cfg.num_search_samples,
                        sigma_weight=sigma_weight,
                        guaranteed_searches=both_task_input_data)

                    tried_gains[s_idx, p_idx, i, :] = best_gain.detach().cpu().numpy()
                    best_gain_unscaled = np.squeeze(
                        gain_scaler.inverse_transform(best_gain.detach().cpu().numpy().reshape(1, -1)))
                    perf_metrics, traj, env = robot.evaluate_x(best_gain_unscaled, env, render=False)
                    perf_metrics[..., kf_cfg.metric_idxs_to_invert] = 1 / (
                                1 + perf_metrics[..., kf_cfg.metric_idxs_to_invert])
                    raw_metrics[s_idx, p_idx, i] = perf_metrics[kf_cfg.metric_idxs]
                    true_y_scaled = np.dot(cost_weights, metric_scaler.transform(
                        perf_metrics[kf_cfg.metric_idxs].reshape(1, -1)).squeeze())

                    expected_costs[s_idx, p_idx, i] = best_y
                    actual_costs[s_idx, p_idx, i] = true_y_scaled
                    actual_scaled_metrics[s_idx, p_idx, i] = metric_scaler.transform(
                        perf_metrics[kf_cfg.metric_idxs].reshape(1, -1)).squeeze()

                    task_input_data[i] = torch.cat([best_gain, track_ahead_pp_torch, torch.from_numpy(traj_flat)], dim=-1)
                    task_target_data[i] = true_y_scaled
                    sigma_weight += experiment_cfg.kf_sigma_step

                    if i > experiment_cfg.lkbo_num_explore_iters-1:
                        gp_model, mll = train_dk_gp(task_input_data[:i+1],
                                                    task_target_data[:i+1].squeeze(),
                                                    kf_network,
                                                    experiment_cfg.lkbo_num_train_iters,
                                                    len(kf_cfg.metric_idxs),
                                                    device)
                        gp_model.eval()
    finally:
        env.close()

    result_arrays = {"expected_costs": expected_costs,
                     "actual_costs": actual_costs,
                     "raw_metrics": raw_metrics,
                     "tried_gains": tried_gains,
                     "actual_scaled_metrics": actual_scaled_metrics,
                     "track_segments": track_segments}
    other_data = {"failed_count":failed_count}
    return result_arrays, other_data 
