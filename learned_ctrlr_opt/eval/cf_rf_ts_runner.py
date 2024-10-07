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

from learned_ctrlr_opt.meta_learning.teacher_student import *
from learned_ctrlr_opt.meta_learning.utils import get_sysid_history_dataset, get_scalers
from learned_ctrlr_opt.opt.optimize_thru_network import *
from learned_ctrlr_opt.opt.random_search import *
from learned_ctrlr_opt.utils.experiment_utils import *
from learned_ctrlr_opt.systems.quadrotor_geom import *
from learned_ctrlr_opt.eval.eval_utils import *


def cf_rf_ts_runner(experiment_cfg, random_seed):
    ts_checkpoint_dir = experiment_cfg.ts_ckpt_dir
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ts_cfg = OmegaConf.load(os.path.join(ts_checkpoint_dir, "config.yaml"))
    gain_dim = len(ts_cfg.gains_to_optimize)
    history_in_size = ts_cfg.history_length * ts_cfg.traj_dim

    ts_network, gain_scaler, history_scaler, metric_scaler = load_ts_and_scalers(experiment_cfg)
    ts_network.use_teacher = experiment_cfg.use_teacher
    _, _, _, metric_scaler = load_kf_and_scalers(experiment_cfg)  # USE KF SCALER FOR CONSISTENCY AMONGST DATASETS

    dset_f = h5py.File(ts_cfg.path_to_dataset, 'r')
    if experiment_cfg.ts_lookahead_dim > 0:
        ref_tracks_enc = np.array(dset_f["reference_tracks_enc"])
        ref_track_scaler = MinMaxScaler(clip=True).fit(ref_tracks_enc.reshape(-1, ref_tracks_enc.shape[-1]))
    thetas = np.array(dset_f["intrinsics"])
    theta_scaler = MinMaxScaler(clip=False).fit(thetas.reshape(-1, thetas.shape[-1]))
    dset_f.close()

    test_set = experiment_cfg.test_set_name
    robot_name = experiment_cfg.robot_name
    params_array, tasks, robot_kwargs = load_test_set(robot_name, test_set)
    params_obj_array = [CrazyFlieParams(*params_array[i]) for i in range(params_array.shape[0])]
    trajs = [ThreeDCircularTraj_fixed(radius=tasks[0, i], freq=tasks[2, i], yaw_bool=False, center=tasks[1, i]) for i in
             range(tasks.shape[1])]

    traj_len = robot_kwargs["t_f"] * 100
    cost_weights = np.array(experiment_cfg.cost_weights)
    initial_gain = np.array(experiment_cfg.initial_gain)
    num_trials = experiment_cfg.num_trials

    expected_costs = np.zeros((len(trajs), len(params_array), num_trials, 1))
    actual_costs = np.zeros((len(trajs), len(params_array), num_trials, 1))
    raw_metrics = np.zeros((len(trajs), len(params_array), num_trials, len(ts_cfg.metric_idxs)))
    full_traj = np.zeros((len(trajs), len(params_array), traj_len * num_trials, ts_cfg.traj_dim))
    tried_gains = np.zeros((len(trajs), len(params_array), num_trials, gain_dim))
    student_latents = np.zeros((len(trajs), len(params_array), num_trials, ts_cfg.latent_dim))
    teacher_latents = np.zeros((len(trajs), len(params_array), num_trials, ts_cfg.latent_dim))
    expected_scaled_metrics = np.zeros((len(trajs), len(params_array), num_trials, len(ts_cfg.metric_idxs)))
    actual_scaled_metrics = np.zeros((len(trajs), len(params_array), num_trials, len(ts_cfg.metric_idxs)))
    failed_count = 0

    for t_idx, traj_obj in enumerate(trajs):
        if "wind" in robot_kwargs:
            wind_obj = ConstantWind(*robot_kwargs["wind"][t_idx])
        else:
            wind_obj = None
        if experiment_cfg.ts_lookahead_dim > 0:
            task = tasks[t_idx].ravel()
            task_enc = torch.from_numpy(ref_track_scaler.transform(task.reshape(1, -1))).squeeze()
        for p_idx, params in enumerate(params_obj_array):
            print(f"seed {random_seed}: ON p_idx {p_idx}")
            success = False
            tried_times = 0
            while not success and tried_times < 10:
                traj_obj.reset()
                robot = QuadrotorSE3Control_ResetFree(params,
                                                      ts_cfg.gains_to_optimize,
                                                      traj_obj,
                                                      robot_kwargs["t_f"])
                metrics, traj, env = robot.evaluate_x(initial_gain, render=False, wind=wind_obj, raw_pos=False)  # no wind for now.
                success = eval_success(env["state"]["x"])
                tried_times += 1
            if not success:
                print("Could not find an initial gain that worked!")
                continue

            if experiment_cfg.use_teacher:
                task_input_data = torch.zeros(num_trials, gain_dim + experiment_cfg.ts_lookahead_dim + ts_cfg.traj_dim + len(ts_cfg.thetas_randomized))
            else:
                task_input_data = torch.zeros(num_trials, gain_dim + experiment_cfg.ts_lookahead_dim + history_in_size)
            task_target_data = torch.zeros(num_trials, len(ts_cfg.metric_idxs))
            theta_enc = torch.from_numpy(theta_scaler.transform(params_array[p_idx].reshape(1, -1))).squeeze()
            np.random.seed(random_seed+p_idx+t_idx)
            torch.manual_seed(random_seed+p_idx+t_idx)
            for i in range(num_trials):
                def eval_fn(x, cost_weights, sigma_weight):
                    q = x.size(0)
                    cost_weights = torch.from_numpy(cost_weights).float().to(device)
                    ys = ts_network(x.float().to(device))
                    cost_weights_batch = cost_weights.repeat(q).reshape((q, cost_weights.shape[-1])).to(device)
                    mean_losses = torch.sum(ys * cost_weights_batch, dim=-1)
                    variances = torch.zeros(x.size(0), device=device)
                    return mean_losses, ys, variances

                if not eval_success(env["state"]["x"]):
                    print(f"went out of bounds!: {env['state']['x'][-1,0:3]}")
                    failed_count += 1
                    break

                traj_lim = traj[-ts_cfg.history_length:]
                traj_lim = np.expand_dims(traj_lim, 0)
                traj_rs = traj_lim.reshape(-1, traj_lim.shape[-1])
                traj_rs[:,ts_cfg.history_sub_idxs] -= traj_rs[0,ts_cfg.history_sub_idxs]
                traj_scaled = history_scaler.transform(traj_rs)
                traj_flat = np.squeeze(traj_scaled.reshape(traj_lim.shape[0], -1))
                l = min(traj_len, env["state"]["x"].shape[0])
                full_traj[t_idx, p_idx, i * l:(i + 1) * l, 0:3] = env["state"]["x"][:l]
                if experiment_cfg.ts_lookahead_dim > 0:
                    if experiment_cfg.use_teacher:
                        fixed_inputs = torch.cat([task_enc, torch.from_numpy(traj_flat[-9:]), theta_enc], dim=-1)
                    else:
                        fixed_inputs = torch.cat([task_enc, torch.from_numpy(traj_flat)], dim=-1)
                else:
                    if experiment_cfg.use_teacher:
                        fixed_inputs = torch.cat([torch.from_numpy(traj_flat[-9:]), theta_enc], dim=-1)
                    else:
                        fixed_inputs = torch.cat([torch.from_numpy(traj_flat)], dim=-1)

                if i > 5:
                    perturbed_optima = [task_input_data[:i, :gain_dim]]
                    for num_to_add in range(10):
                        noisy_task_input_data = torch.clip(
                            task_input_data[:i, :gain_dim] + torch.randn(task_input_data[:i, :gain_dim].shape) * 3e-2,
                            0, 1)
                        perturbed_optima.append(noisy_task_input_data)
                    both_task_input_data = torch.cat(perturbed_optima, dim=0)
                else:
                    both_task_input_data = task_input_data[:i, :gain_dim]
                if i == 0:
                    best_gain, best_cost, best_y, all_tested_gains, all_tested_ys, all_tested_costs = random_search(eval_fn,
                                                                                                                    experiment_cfg.num_search_samples,
                                                                                                                    cost_weights,
                                                                                                                    gain_dim,
                                                                                                                    device,
                                                                                                                    fixed_inputs,
                                                                                                                    batch_size=512,
                                                                                                                    sigma_weight=0,
                                                                                                                    guaranteed_searches=both_task_input_data[
                                                                                                                                        :i,
                                                                                                                                        :gain_dim])
                tried_gains[t_idx, p_idx, i, :] = best_gain.detach().cpu().numpy()
                best_gain_unscaled = np.squeeze(
                    gain_scaler.inverse_transform(best_gain.detach().cpu().numpy().reshape(1, -1)))
                # print(f"---- Step {i} ----")
                # print(f"best_gain_unscaled = {best_gain_unscaled}")
                perf_metrics, traj, env = robot.evaluate_x(best_gain_unscaled, env, render=False, wind=wind_obj, raw_pos=False)
                # print(f"perf metrics were {perf_metrics}")
                perf_metrics[..., ts_cfg.metric_idxs_to_invert] = 1 / (
                        1 + perf_metrics[..., ts_cfg.metric_idxs_to_invert])
                raw_metrics[t_idx, p_idx, i] = perf_metrics
                # print(
                #     f"scaled perf metrics were {metric_scaler.transform(perf_metrics[ts_cfg.metric_idxs].reshape(1, -1)).squeeze()}")
                true_y_scaled = np.dot(cost_weights, metric_scaler.transform(
                    perf_metrics[ts_cfg.metric_idxs].reshape(1, -1)).squeeze())

                expected_costs[t_idx, p_idx, i] = best_y
                actual_costs[t_idx, p_idx, i] = true_y_scaled
                actual_scaled_metrics[t_idx, p_idx, i] = metric_scaler.transform(
                    perf_metrics[ts_cfg.metric_idxs].reshape(1, -1)).squeeze()

                input_test = torch.cat([best_gain, fixed_inputs], dim=-1).unsqueeze(0)
                best_y_mean = ts_network(input_test.float().to(device))
                latent = ts_network.encode_student(torch.from_numpy(traj_flat).unsqueeze(0).float().to(device))
                student_latents[t_idx, p_idx, i] = latent.cpu().detach().numpy()
                teacher_latents[t_idx, p_idx, i] = ts_network.encode_teacher(torch.cat([torch.from_numpy(traj_flat[-9:]), theta_enc], dim=-1).float().to(device).unsqueeze(0)).cpu().detach().numpy()
                expected_scaled_metrics[t_idx, p_idx, i] = best_y_mean.cpu().detach().numpy()

                target = torch.from_numpy(
                    metric_scaler.transform(perf_metrics[ts_cfg.metric_idxs].reshape(1, -1)).squeeze())
                task_input_data[i] = input_test
                task_target_data[i] = target

    result_arrays = {"expected_costs": expected_costs,
                     "actual_costs": actual_costs,
                     "raw_metrics": raw_metrics,
                     "tried_gains": tried_gains,
                     "expected_scaled_metrics": expected_scaled_metrics,
                     "actual_scaled_metrics": actual_scaled_metrics,
                     "full_traj": full_traj,
                     "student_latents": student_latents,
                     "teacher_latents": teacher_latents}
    other_data = {"failed_count": failed_count, "used_teacher": experiment_cfg.use_teacher}
    return result_arrays, other_data
