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

from learned_ctrlr_opt.meta_learning.lsr_net import *
from learned_ctrlr_opt.meta_learning.basis_kf import *
from learned_ctrlr_opt.meta_learning.utils import get_sysid_history_dataset, get_scalers
from learned_ctrlr_opt.opt.random_search import *
from learned_ctrlr_opt.opt.particle_search import *
from learned_ctrlr_opt.systems.car_controller import pp_track_curvature
from learned_ctrlr_opt.systems.car_dynamics import CarParams
from learned_ctrlr_opt.systems.robots import TopDownCarResetFree
from learned_ctrlr_opt.utils.experiment_utils import *
from learned_ctrlr_opt.eval.eval_utils import load_kf_and_scalers


def eval_success(traj):
    return np.abs(traj[-1][2]) < 15


def tdc_rf_kf_runner(experiment_cfg, random_seed, adapt=True):
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
    sigma_weight_start = experiment_cfg.kf_sigma_penalty
    initial_gain = np.array(experiment_cfg.initial_gain)
    num_trials = experiment_cfg.num_trials

    expected_costs = np.zeros((len(seeds), len(params_array), num_trials, 1))
    expected_variances = np.zeros((len(seeds), len(params_array), num_trials, 1))
    actual_costs = np.zeros((len(seeds), len(params_array), num_trials, 1))
    raw_metrics = np.zeros((len(seeds), len(params_array), num_trials, len(kf_cfg.metric_idxs)))
    tried_gains = np.zeros((len(seeds), len(params_array), num_trials, len(kf_cfg.gains_to_optimize)))
    all_weights = np.zeros((len(seeds), len(params_array), num_trials, kf_cfg.n_bases))
    expected_scaled_metrics = np.zeros((len(seeds), len(params_array), num_trials, len(kf_cfg.metric_idxs)))
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

                if experiment_cfg.no_meta:
                    kf_network.initialize_priors()

                weights = kf_network.last_layer_prior
                sigma = torch.mm(kf_network.last_layer_prior_cov_sqrt, torch.t(kf_network.last_layer_prior_cov_sqrt)).float().to(device)
                Q = torch.mm(kf_network.Q_sqrt, torch.t(kf_network.Q_sqrt)).float().to(device)
                R = torch.mm(kf_network.R_sqrt, torch.t(kf_network.R_sqrt)).float().to(device)

                task_input_data = torch.zeros(num_trials, gain_dim + lookahead_dim + history_in_size)
                task_target_data = torch.zeros(num_trials, len(kf_cfg.metric_idxs))
                sigma_weight = sigma_weight_start
                np.random.seed(random_seed+p_idx+s_idx)
                torch.manual_seed(random_seed+p_idx+s_idx)
                for i in range(num_trials):
                    all_weights[s_idx, p_idx, i] = weights.cpu().detach().numpy()
                    def eval_fn(x, cost_weights, sigma_weight):
                        q = x.size(0)
                        cost_weights = torch.from_numpy(cost_weights).float().to(device)
                        ys, sigmas = last_layer_prediction_uncertainty_aware(x.float().to(device), kf_network, weights, sigma)
                        cost_weights_batch = cost_weights.repeat(q).reshape((q, cost_weights.shape[-1])).to(device)
                        mean_losses = torch.sum(ys * cost_weights_batch, dim=-1)
                        variances = torch.zeros(x.size(0)).to(device)
                        for j in range(x.size(0)):
                            inter = torch.mm(sigmas[j], cost_weights.unsqueeze(-1))
                            variances[j] = torch.mm(torch.t(cost_weights.unsqueeze(-1)), inter)
                        losses = mean_losses + sigma_weight * variances
                        return losses, ys, variances

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

                    best_gain, best_cost, best_y, all_tested_gains, all_tested_ys, all_tested_costs = random_search(
                        eval_fn,
                        experiment_cfg.num_search_samples,
                        cost_weights,
                        gain_dim,
                        device,
                        torch.cat([track_ahead_pp_torch, torch.from_numpy(traj_flat)]),
                        batch_size=512,
                        sigma_weight=sigma_weight,
                        # guaranteed_searches=None)
                        guaranteed_searches=both_task_input_data)

                    # should I save unscaled or scaled gains?
                    tried_gains[s_idx, p_idx, i, :] = best_gain.detach().cpu().numpy()
                    best_gain_unscaled = np.squeeze(
                        gain_scaler.inverse_transform(best_gain.detach().cpu().numpy().reshape(1, -1)))
                    # print(f"---- Step {i} ----")
                    # print(f"best_gain_unscaled = {best_gain_unscaled}")
                    perf_metrics, traj, env = robot.evaluate_x(best_gain_unscaled, env, render=False)
                    perf_metrics[..., kf_cfg.metric_idxs_to_invert] = 1 / (
                                1 + perf_metrics[..., kf_cfg.metric_idxs_to_invert])
                    raw_metrics[s_idx, p_idx, i] = perf_metrics[kf_cfg.metric_idxs]
                    # print(f"perf metrics were {perf_metrics}")
                    # print(
                    #     f"scaled perf metrics were {metric_scaler.transform(perf_metrics[kf_cfg.metric_idxs].reshape(1, -1)).squeeze()}")
                    true_y_scaled = np.dot(cost_weights, metric_scaler.transform(
                        perf_metrics[kf_cfg.metric_idxs].reshape(1, -1)).squeeze())

                    expected_costs[s_idx, p_idx, i] = best_y
                    actual_costs[s_idx, p_idx, i] = true_y_scaled
                    actual_scaled_metrics[s_idx, p_idx, i] = metric_scaler.transform(
                        perf_metrics[kf_cfg.metric_idxs].reshape(1, -1)).squeeze()

                    input_test = torch.cat([best_gain, track_ahead_pp_torch, torch.from_numpy(traj_flat)],
                                           dim=-1).unsqueeze(0)
                    best_y_mean, best_y_sigma = last_layer_prediction_uncertainty_aware(input_test, kf_network, weights,
                                                                                        sigma)
                    expected_scaled_metrics[s_idx, p_idx, i] = best_y_mean.cpu().detach().numpy()
                    expected_variances[
                        s_idx, p_idx, i] = cost_weights.T @ best_y_sigma.cpu().detach().numpy() @ cost_weights

                    inp = torch.cat([best_gain, track_ahead_pp_torch, torch.from_numpy(traj_flat)], dim=-1)
                    phi = kf_network(inp.unsqueeze(0).float().to(device)).squeeze().detach()
                    target = torch.from_numpy(
                        metric_scaler.transform(perf_metrics[kf_cfg.metric_idxs].reshape(1, -1)).squeeze())
                    task_input_data[i] = inp
                    task_target_data[i] = target
                    if adapt:
                        with torch.no_grad():
                            weights, sigma, K = kalman_step(weights, sigma, target.float().to(device), phi, Q, R)

                    # Replay Buffer
                    if i > 0:
                        for k in range(experiment_cfg.num_replay_steps):
                            replay_idx = np.random.randint(0, i)
                            phi = kf_network(
                                task_input_data[replay_idx].unsqueeze(0).float().to(device)).squeeze().detach()
                            target = task_target_data[replay_idx]
                            with torch.no_grad():
                                weights, sigma, K = kalman_step(weights, sigma, target.float().to(device), phi, Q, R)
                    sigma_weight += experiment_cfg.kf_sigma_step
    finally:
        env.close()

    result_arrays = {"expected_costs": expected_costs,
                     "expected_variances" :expected_variances,
                     "actual_costs": actual_costs,
                     "raw_metrics": raw_metrics,
                     "tried_gains": tried_gains,
                     "all_weights": all_weights,
                     "expected_scaled_metrics": expected_scaled_metrics,
                     "actual_scaled_metrics": actual_scaled_metrics,
                     "track_segments": track_segments}
    other_data = {"failed_count":failed_count}
    return result_arrays, other_data


def tdc_rf_kf_runner_idx(experiment_cfg,
                         random_seed,
                         track_idx,
                         param_idx,
                         adapt=True):
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
    sigma_weight_start = experiment_cfg.kf_sigma_penalty
    initial_gain = np.array(experiment_cfg.initial_gain)
    num_trials = experiment_cfg.num_trials

    failed_count = 0
    seed = seeds[track_idx]
    params = params_array[param_idx]
    print(params)
    np.random.seed(random_seed+param_idx+track_idx)
    torch.manual_seed(random_seed+param_idx+track_idx)

    try:
        robot = TopDownCarResetFree(seed=int(seed),
                                    car_params=params,
                                    gains_to_optimize=kf_cfg.gains_to_optimize,
                                    length=experiment_cfg.segment_length,
                                    max_time=robot_kwargs["max_time"],
                                    record=True)


        success = False
        metrics, traj, env = robot.evaluate_x(initial_gain,
                                              render=True,
                                              vid_folder="videos/",
                                              vid_prefix=f"tdc_kf_{random_seed}_{track_idx}_{param_idx}")
        success = eval_success(traj)
        if not success:
            print("Could not find an initial gain that worked!")
            return

        # env = gym.wrappers.RecordVideo(env=env, video_folder="videos/", name_prefix='test', step_trigger=lambda x: x==0)
        # env.start_video_recorder()

        if experiment_cfg.no_meta:
            kf_network.initialize_priors()

        weights = kf_network.last_layer_prior
        sigma = torch.mm(kf_network.last_layer_prior_cov_sqrt, torch.t(kf_network.last_layer_prior_cov_sqrt)).float().to(device)
        Q = torch.mm(kf_network.Q_sqrt, torch.t(kf_network.Q_sqrt)).float().to(device)
        R = torch.mm(kf_network.R_sqrt, torch.t(kf_network.R_sqrt)).float().to(device)

        task_input_data = torch.zeros(num_trials, gain_dim + lookahead_dim + history_in_size)
        task_target_data = torch.zeros(num_trials, len(kf_cfg.metric_idxs))
        sigma_weight = sigma_weight_start
        for i in range(num_trials):
            def eval_fn(x, cost_weights, sigma_weight):
                q = x.size(0)
                cost_weights = torch.from_numpy(cost_weights).float().to(device)
                ys, sigmas = last_layer_prediction_uncertainty_aware(x.float().to(device), kf_network, weights, sigma)
                cost_weights_batch = cost_weights.repeat(q).reshape((q, cost_weights.shape[-1])).to(device)
                mean_losses = torch.sum(ys * cost_weights_batch, dim=-1)
                variances = torch.zeros(x.size(0)).to(device)
                for j in range(x.size(0)):
                    inter = torch.mm(sigmas[j], cost_weights.unsqueeze(-1))
                    variances[j] = torch.mm(torch.t(cost_weights.unsqueeze(-1)), inter)
                losses = mean_losses + sigma_weight * variances
                return losses, ys, variances

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

            best_gain, best_cost, best_y, all_tested_gains, all_tested_ys, all_tested_costs = random_search(
                eval_fn,
                experiment_cfg.num_search_samples,
                cost_weights,
                gain_dim,
                device,
                torch.cat([track_ahead_pp_torch, torch.from_numpy(traj_flat)]),
                batch_size=512,
                sigma_weight=sigma_weight,
                # guaranteed_searches=None)
                guaranteed_searches=both_task_input_data)

            # should I save unscaled or scaled gains?
            best_gain_unscaled = np.squeeze(
                gain_scaler.inverse_transform(best_gain.detach().cpu().numpy().reshape(1, -1)))
            # print(f"---- Step {i} ----")
            # print(f"best_gain_unscaled = {best_gain_unscaled}")
            perf_metrics, traj, env = robot.evaluate_x(best_gain_unscaled, env, render=False)
            perf_metrics[..., kf_cfg.metric_idxs_to_invert] = 1 / (
                    1 + perf_metrics[..., kf_cfg.metric_idxs_to_invert])
            # print(f"perf metrics were {perf_metrics}")
            # print(
            #     f"scaled perf metrics were {metric_scaler.transform(perf_metrics[kf_cfg.metric_idxs].reshape(1, -1)).squeeze()}")
            true_y_scaled = np.dot(cost_weights, metric_scaler.transform(
                perf_metrics[kf_cfg.metric_idxs].reshape(1, -1)).squeeze())

            input_test = torch.cat([best_gain, track_ahead_pp_torch, torch.from_numpy(traj_flat)],
                                   dim=-1).unsqueeze(0)
            best_y_mean, best_y_sigma = last_layer_prediction_uncertainty_aware(input_test, kf_network, weights,
                                                                                sigma)

            inp = torch.cat([best_gain, track_ahead_pp_torch, torch.from_numpy(traj_flat)], dim=-1)
            phi = kf_network(inp.unsqueeze(0).float().to(device)).squeeze().detach()
            target = torch.from_numpy(
                metric_scaler.transform(perf_metrics[kf_cfg.metric_idxs].reshape(1, -1)).squeeze())
            task_input_data[i] = inp
            task_target_data[i] = target
            if adapt:
                with torch.no_grad():
                    weights, sigma, K = kalman_step(weights, sigma, target.float().to(device), phi, Q, R)

            # Replay Buffer
            if i > 0:
                for k in range(experiment_cfg.num_replay_steps):
                    replay_idx = np.random.randint(0, i)
                    phi = kf_network(
                        task_input_data[replay_idx].unsqueeze(0).float().to(device)).squeeze().detach()
                    target = task_target_data[replay_idx]
                    with torch.no_grad():
                        weights, sigma, K = kalman_step(weights, sigma, target.float().to(device), phi, Q, R)
            sigma_weight += experiment_cfg.kf_sigma_step
    finally:
        env.close_video_recorder()
        env.close()
