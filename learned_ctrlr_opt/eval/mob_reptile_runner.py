from learned_ctrlr_opt.systems.mob_locomotion import *
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
from learned_ctrlr_opt.meta_learning.utils import get_sysid_history_dataset, get_scalers
from learned_ctrlr_opt.opt.optimize_thru_network import *
from learned_ctrlr_opt.opt.random_search import *
from learned_ctrlr_opt.utils.experiment_utils import *
from learned_ctrlr_opt.eval.eval_utils import *
from learned_ctrlr_opt.utils.dataset_utils import normalize


def mob_reptile_runner(experiment_cfg, random_seeds, adapt=True):
    reptile_checkpoint_dir = experiment_cfg.reptile_ckpt_dir
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    reptile_cfg = OmegaConf.load(os.path.join(reptile_checkpoint_dir, "config.yaml"))
    gain_dim = len(reptile_cfg.gains_to_optimize)
    history_in_size = reptile_cfg.history_length * reptile_cfg.traj_dim
    lookahead_dim = experiment_cfg.lookahead_dim

    reptile_network, gain_scaler, history_scaler, metric_scaler = load_reptile_and_scalers(experiment_cfg)

    test_set = experiment_cfg.test_set_name
    robot_name = experiment_cfg.robot_name
    params_array, tasks, robot_kwargs = load_test_set(robot_name, test_set)
    # params_array = [Go1HiddenParams(*params_array[i]) for i in range(params_array.shape[0])]
    cmds = tasks

    traj_len = robot_kwargs["eval_length_s"] * 50
    cost_weights = np.array(experiment_cfg.cost_weights)
    sigma_weight_start = experiment_cfg.kf_sigma_penalty
    num_trials = experiment_cfg.num_trials

    dset_f = h5py.File(reptile_cfg.path_to_dataset, 'r')
    ref_tracks_enc = np.array(dset_f["reference_tracks_enc"])
    ref_track_scaler = MinMaxScaler(clip=True).fit(ref_tracks_enc.reshape(-1, lookahead_dim))
    dset_f.close()

    assert num_trials == cmds.shape[1]

    expected_costs = np.zeros((len(random_seeds), len(cmds), len(params_array), num_trials, 1))
    actual_costs = np.zeros((len(random_seeds), len(cmds), len(params_array), num_trials, 1))
    raw_metrics = np.zeros((len(random_seeds), len(cmds), len(params_array), num_trials, len(reptile_cfg.metric_idxs)))
    full_traj = np.zeros((len(random_seeds), len(cmds), len(params_array), traj_len * num_trials, reptile_cfg.traj_dim))
    tried_gains = np.zeros((len(random_seeds), len(cmds), len(params_array), num_trials, gain_dim))
    expected_scaled_metrics = np.zeros((len(random_seeds), len(cmds), len(params_array), num_trials, len(reptile_cfg.metric_idxs)))
    actual_scaled_metrics = np.zeros((len(random_seeds), len(cmds), len(params_array), num_trials, len(reptile_cfg.metric_idxs)))
    failed_counts = np.zeros(len(random_seeds))

    n_envs = params_array.shape[0]
    robot = MoBLocomotion_ResetFree(gains_to_optimize=reptile_cfg.gains_to_optimize,
                                    params=params_array,
                                    num_envs=n_envs,
                                    render=False,
                                    eval_length_s=robot_kwargs["eval_length_s"])
    initial_gains = np.array([experiment_cfg.initial_gain for w in range(n_envs)])
    # every random seed is evaluated on every (task, param) combination
    gaits_torch = torch.tensor(GAITS_NORM, dtype=torch.float32, device=device)
    def mob_sampler(q):
        samples = torch.rand(q, gain_dim, device=device)
        if 2 in reptile_cfg.gains_to_optimize and 3 in reptile_cfg.gains_to_optimize and 4 in reptile_cfg.gains_to_optimize:
            gait_idxs = torch.randint(0, gaits_torch.size(0), size=(q,), device=device)
            samples[:,[2,3,4]] = gaits_torch[gait_idxs]
        return samples

    for s_idx, seed in enumerate(random_seeds):
        print(f"---- On seed {seed} ----")
        for t_idx, cmd in enumerate(cmds):
            print(f"on t idx {t_idx}")
            np.random.seed(seed + t_idx + 10000)
            torch.manual_seed(seed + t_idx + 10000)
            robot.reset_env()

            cmd_rpt = np.zeros((n_envs, 3))
            cmd_rpt[:,:] = cmds[t_idx,0,:]
            metrics, obtained_trajs, successes = robot.evaluate_x(initial_gains, cmd=cmd_rpt)
            task_input_data = torch.zeros(n_envs, num_trials, gain_dim + lookahead_dim + history_in_size)
            task_target_data = torch.zeros(n_envs, num_trials, len(reptile_cfg.metric_idxs))
            success_mask = np.array(successes, dtype=bool)
            for i in range(num_trials):
                print(f"On trial {i}")
                gains_to_try = []
                for p_idx in range(params_array.shape[0]):
                    if not successes[p_idx]:
                        if success_mask[p_idx]:
                            failed_counts[s_idx] += 1
                            success_mask[p_idx] = False
                        gains_to_try.append(initial_gains[0])  # just do the initial gain, but stop adapting and collecting perf metrics?
                        continue

                    if i == 0:
                        adapted_net = reptile_network
                    else:
                        # adapt reptile to data here
                        adapted_net = adapt_reptile(task_input_data[p_idx,:i],
                                                    task_target_data[p_idx,:i],
                                                    reptile_network,
                                                    reptile_cfg.inner_lr,
                                                    reptile_cfg.num_inner_steps)

                    def eval_fn(x, cost_weights, sigma_weight):
                        q = x.size(0)
                        cost_weights = torch.from_numpy(cost_weights)
                        ys = adapted_net(x.float())
                        cost_weights_batch = cost_weights.repeat(q).reshape((q, cost_weights.shape[-1])).to(device)
                        mean_losses = torch.sum(ys * cost_weights_batch, dim=-1)
                        losses = mean_losses
                        return losses, ys, torch.zeros(*losses.shape)

                    traj_lim = obtained_trajs[p_idx, -reptile_cfg.history_length:]
                    traj_lim = np.expand_dims(traj_lim, 0)
                    traj_rs = traj_lim.reshape(-1, traj_lim.shape[-1])
                    traj_scaled = history_scaler.transform(traj_rs)
                    traj_flat = np.squeeze(traj_scaled.reshape(traj_lim.shape[0], -1))
                    cmd_scaled = ref_track_scaler.transform(cmds[t_idx, i].reshape(1, -1)).squeeze()
                    if i > 0:
                        perturbed_optima = [task_input_data[p_idx, :i, :gain_dim]]
                        for num_to_add in range(20):
                            noises = torch.randn(task_input_data[p_idx, :i, :gain_dim].shape) * 5e-2
                            noises[:, [2, 3, 4]] = 0
                            noisy_task_input_data = torch.clip(task_input_data[p_idx, :i, :gain_dim] + noises, 0, 1)
                            perturbed_optima.append(noisy_task_input_data)
                        both_task_input_data = torch.cat(perturbed_optima, dim=0)
                    else:
                        both_task_input_data = None
                    best_gain, best_cost, best_y, all_tested_gains, all_tested_ys, all_tested_costs = random_search(
                        eval_fn,
                        experiment_cfg.num_search_samples,
                        cost_weights,
                        gain_dim,
                        device,
                        torch.cat([torch.from_numpy(cmd_scaled), torch.from_numpy(traj_flat)]),
                        batch_size=experiment_cfg.num_search_samples,
                        sigma_weight=0.0,
                        guaranteed_searches=both_task_input_data,
                        sampler=mob_sampler)

                    tried_gains[s_idx, t_idx, p_idx,i] = best_gain.numpy()
                    expected_costs[s_idx, t_idx, p_idx, i] = best_y
                    gains_to_try.append(best_gain.numpy())
                    input_test = torch.cat([best_gain, torch.from_numpy(cmd_scaled), torch.from_numpy(traj_flat)],
                                           dim=-1).unsqueeze(0)
                    expected_scaled_metrics[s_idx, t_idx, p_idx, i] = adapted_net(input_test.unsqueeze(0).float().to(device)).cpu().detach().numpy().squeeze()
                    task_input_data[p_idx, i] = input_test
                    full_traj[s_idx, t_idx, p_idx, i*traj_len:(i+1)*traj_len] = obtained_trajs[p_idx]

                # Once all gains are obtained, run the simulation
                gains_unscaled = gain_scaler.inverse_transform(np.array(gains_to_try))
                cmd_rpt = np.zeros((n_envs, 3))
                cmd_rpt[:,:] = cmds[t_idx,i,:]
                perf_metrics, obtained_trajs, successes = robot.evaluate_x(gains_unscaled, cmd_rpt)
                perf_metrics[...,reptile_cfg.metric_idxs_to_invert] = 1/(1+perf_metrics[...,reptile_cfg.metric_idxs_to_invert])
                perf_metrics_scaled = metric_scaler.transform(perf_metrics[...,reptile_cfg.metric_idxs])
                true_y_scaled = np.dot(perf_metrics_scaled, cost_weights)
                actual_costs[s_idx, t_idx, success_mask, i] = true_y_scaled.reshape((-1, 1))[success_mask]

                raw_metrics[s_idx,t_idx,success_mask,i] = perf_metrics[success_mask][:,reptile_cfg.metric_idxs]
                actual_scaled_metrics[s_idx,t_idx,success_mask,i] = perf_metrics_scaled[success_mask]
                task_target_data[:,i] = torch.from_numpy(perf_metrics_scaled)

    results = []
    for s_idx, seed in enumerate(random_seeds):
        result = ({"expected_costs": expected_costs[s_idx],
                   "actual_costs": actual_costs[s_idx],
                   "raw_metrics": raw_metrics[s_idx],
                   "tried_gains": tried_gains[s_idx],
                   "expected_scaled_metrics": expected_scaled_metrics[s_idx],
                   "actual_scaled_metrics": actual_scaled_metrics[s_idx],
                   "full_traj": full_traj[s_idx]})
        results.append((result, {"failed_count":failed_counts[s_idx]}))
    return results
