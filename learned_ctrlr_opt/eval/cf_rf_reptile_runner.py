import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch.optim
from rotorpy.wind.default_winds import ConstantWind
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
from learned_ctrlr_opt.opt.optimize_thru_network import *
from learned_ctrlr_opt.opt.random_search import *
from learned_ctrlr_opt.utils.experiment_utils import *
from learned_ctrlr_opt.systems.quadrotor_geom import *
from learned_ctrlr_opt.eval.eval_utils import *


def cf_rf_reptile_runner(experiment_cfg, random_seed):
    reptile_checkpoint_dir = experiment_cfg.reptile_ckpt_dir
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    reptile_cfg = OmegaConf.load(os.path.join(reptile_checkpoint_dir, "config.yaml"))
    gain_dim = len(reptile_cfg.gains_to_optimize)
    history_in_size = reptile_cfg.history_length * reptile_cfg.traj_dim

    reptile_net, gain_scaler, history_scaler, metric_scaler = load_reptile_and_scalers(experiment_cfg)

    test_set = experiment_cfg.test_set_name
    robot_name = experiment_cfg.robot_name
    params_array, tasks, robot_kwargs = load_test_set(robot_name, test_set)
    params_array = [CrazyFlieParams(*params_array[i]) for i in range(params_array.shape[0])]
    trajs = [ThreeDCircularTraj_fixed(radius=tasks[0, i], freq=tasks[2, i], yaw_bool=False, center=tasks[1, i]) for i in
             range(tasks.shape[1])]

    traj_len = robot_kwargs["t_f"] * 100
    cost_weights = np.array(experiment_cfg.cost_weights)
    sigma_weight_start = experiment_cfg.kf_sigma_penalty
    initial_gain = np.array(experiment_cfg.initial_gain)
    num_trials = experiment_cfg.num_trials

    expected_costs = np.zeros((len(trajs), len(params_array), num_trials, 1))
    actual_costs = np.zeros((len(trajs), len(params_array), num_trials, 1))
    raw_metrics = np.zeros((len(trajs), len(params_array), num_trials, len(reptile_cfg.metric_idxs)))
    full_traj = np.zeros((len(trajs), len(params_array), traj_len * num_trials, reptile_cfg.traj_dim))
    tried_gains = np.zeros((len(trajs), len(params_array), num_trials, gain_dim))
    expected_scaled_metrics = np.zeros((len(trajs), len(params_array), num_trials, len(reptile_cfg.metric_idxs)))
    actual_scaled_metrics = np.zeros((len(trajs), len(params_array), num_trials, len(reptile_cfg.metric_idxs)))
    failed_count = 0

    for t_idx, traj_obj in enumerate(trajs):
        if "wind" in robot_kwargs:
            wind_obj = ConstantWind(*robot_kwargs["wind"][t_idx])
        else:
            wind_obj = None
        for p_idx, params in enumerate(params_array):
            print(f"seed {random_seed}: ON p_idx {p_idx}")
            success = False
            tried_times = 0
            while not success and tried_times < 10:
                traj_obj.reset()
                robot = QuadrotorSE3Control_ResetFree(params,
                                                      reptile_cfg.gains_to_optimize,
                                                      traj_obj,
                                                      robot_kwargs["t_f"])
                metrics, traj, env = robot.evaluate_x(initial_gain, render=False, wind=wind_obj)
                success = eval_success(traj)
                tried_times += 1
            if not success:
                print("Could not find an initial gain that worked!")
                continue

            adapted_net = reptile_net
            task_input_data = torch.zeros(num_trials, gain_dim + history_in_size)
            task_target_data = torch.zeros(num_trials, len(reptile_cfg.metric_idxs))
            np.random.seed(random_seed+p_idx+t_idx+10000)  # add a fixed offset here to remove corrleation with kf
            torch.manual_seed(random_seed+p_idx+t_idx+10000)
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
                    print("went out of bounds!")
                    failed_count += 1
                    break

                traj_lim = traj[-reptile_cfg.history_length:]
                traj_lim = np.expand_dims(traj_lim, 0)
                traj_rs = traj_lim.reshape(-1, traj_lim.shape[-1])
                traj_scaled = history_scaler.transform(traj_rs)
                traj_flat = np.squeeze(traj_scaled.reshape(traj_lim.shape[0], -1))
                full_traj[t_idx, p_idx, i * traj_len:(i + 1) * traj_len, :] = traj

                if i > 0:
                    perturbed_optima = [task_input_data[:i, :gain_dim]]
                    for num_to_add in range(10):
                        noisy_task_input_data = torch.clip(
                            task_input_data[:i, :gain_dim] + torch.randn(task_input_data[:i, :gain_dim].shape) * 3e-2,
                            0, 1)
                        perturbed_optima.append(noisy_task_input_data)
                    both_task_input_data = torch.cat(perturbed_optima, dim=0)
                else:
                    both_task_input_data = task_input_data[:i, :gain_dim]
                best_gain, best_cost, best_y, all_tested_gains, all_tested_ys, all_tested_costs = random_search(eval_fn,
                                                                                                                experiment_cfg.num_search_samples,
                                                                                                                cost_weights,
                                                                                                                gain_dim,
                                                                                                                device,
                                                                                                                torch.from_numpy(
                                                                                                                    traj_flat),
                                                                                                                batch_size=512,
                                                                                                                sigma_weight=0,
                                                                                                                guaranteed_searches=both_task_input_data[
                                                                                                                                    :i,
                                                                                                                                    :gain_dim])
                tried_gains[t_idx, p_idx, i, :] = best_gain.detach().cpu().numpy()
                best_gain_unscaled = np.squeeze(
                    gain_scaler.inverse_transform(best_gain.detach().cpu().numpy().reshape(1, -1)))
                print(f"---- Step {i} ----")
                print(f"best_gain_unscaled = {best_gain_unscaled}")
                perf_metrics, traj, env = robot.evaluate_x(best_gain_unscaled, env, render=False, wind=wind_obj)
                print(f"perf metrics were {perf_metrics}")
                perf_metrics[..., reptile_cfg.metric_idxs_to_invert] = 1 / (
                            1 + perf_metrics[..., reptile_cfg.metric_idxs_to_invert])
                raw_metrics[t_idx, p_idx, i] = perf_metrics
                print(
                    f"scaled perf metrics were {metric_scaler.transform(perf_metrics[reptile_cfg.metric_idxs].reshape(1, -1)).squeeze()}")
                true_y_scaled = np.dot(cost_weights, metric_scaler.transform(
                    perf_metrics[reptile_cfg.metric_idxs].reshape(1, -1)).squeeze())

                expected_costs[t_idx, p_idx, i] = best_y
                actual_costs[t_idx, p_idx, i] = true_y_scaled
                actual_scaled_metrics[t_idx, p_idx, i] = metric_scaler.transform(
                    perf_metrics[reptile_cfg.metric_idxs].reshape(1, -1)).squeeze()


                inp = torch.cat([best_gain, torch.from_numpy(traj_flat)], dim=-1)
                expected_scaled_metrics[t_idx, p_idx, i] = adapted_net(inp.unsqueeze(0).float().to(device)).cpu().detach().numpy()
                target = torch.from_numpy(
                    metric_scaler.transform(perf_metrics[reptile_cfg.metric_idxs].reshape(1, -1)).squeeze())
                task_input_data[i] = inp
                task_target_data[i] = target
                adapted_net = adapt_reptile(task_input_data[:i + 1],
                                            task_target_data[:i + 1],
                                            reptile_net,
                                            reptile_cfg.inner_lr,
                                            reptile_cfg.num_inner_steps)

    result_arrays = {"expected_costs": expected_costs,
                     "actual_costs": actual_costs,
                     "raw_metrics": raw_metrics,
                     "tried_gains": tried_gains,
                     "expected_scaled_metrics": expected_scaled_metrics,
                     "actual_scaled_metrics": actual_scaled_metrics,
                     "full_traj": full_traj}
    other_data = {"failed_count": failed_count}
    return result_arrays, other_data
