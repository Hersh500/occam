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

def cf_rf_nom_gain_runner(experiment_cfg, random_seed):
    kf_checkpoint_dir = experiment_cfg.kf_ckpt_dir
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    kf_cfg = OmegaConf.load(os.path.join(kf_checkpoint_dir, "config.yaml"))
    gain_dim = len(kf_cfg.gains_to_optimize)
    history_in_size = kf_cfg.history_length * kf_cfg.traj_dim
    scalers = get_scalers(kf_cfg.path_to_dataset, kf_cfg.history_length, kf_cfg.metric_idxs,
                          kf_cfg.metric_idxs_to_invert)
    history_scaler = scalers[-1]
    gain_scaler = scalers[0]
    metric_scaler = scalers[2]

    test_set = experiment_cfg.test_set_name
    robot_name = experiment_cfg.robot_name
    params_array, tasks, robot_kwargs = load_test_set(robot_name, test_set)
    params_array = [CrazyFlieParams(*params_array[i]) for i in range(params_array.shape[0])]
    trajs = [ThreeDCircularTraj_fixed(radius=tasks[0, i], freq=tasks[2, i], yaw_bool=False, center=tasks[1, i]) for i in
             range(tasks.shape[1])]

    traj_len = robot_kwargs["t_f"] * 100
    cost_weights = np.array(experiment_cfg.cost_weights)
    initial_gain = np.array(experiment_cfg.initial_gain)
    num_trials = experiment_cfg.num_trials

    actual_costs_fixed = np.zeros((len(trajs), len(params_array), num_trials, 1))
    raw_metrics_fixed = np.zeros((len(trajs), len(params_array), num_trials, len(kf_cfg.metric_idxs)))
    actual_scaled_metrics_fixed = np.zeros((len(trajs), len(params_array), num_trials, len(kf_cfg.metric_idxs)))
    full_traj_fixed = np.zeros((len(trajs), len(params_array), traj_len * num_trials, kf_cfg.traj_dim))
    failed_count_ig = 0
    if experiment_cfg.adaptive:
        print(f"Using adaptive control baseline!")

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
                                                      kf_cfg.gains_to_optimize,
                                                      traj_obj,
                                                      robot_kwargs["t_f"])
                metrics, traj, env = robot.evaluate_x(initial_gain, render=False, wind=wind_obj, adaptive=False, adaptive_alpha=experiment_cfg.alpha)
                success = eval_success(traj)
                tried_times += 1
            if not success:
                print("Could not find an initial gain that worked!")
                continue

            np.random.seed(random_seed+p_idx+t_idx)
            torch.manual_seed(random_seed+p_idx+t_idx)
            for i in range(num_trials):
                if not eval_success(traj):
                    print("went out of bounds!")
                    failed_count_ig += 1
                    break

                full_traj_fixed[t_idx, p_idx, i*traj_len:(i+1)*traj_len,:] = traj
                best_gain_unscaled = initial_gain
                perf_metrics, traj, env = robot.evaluate_x(best_gain_unscaled, env, render=False, wind=wind_obj, adaptive=experiment_cfg.adaptive, adaptive_alpha=experiment_cfg.alpha)
                perf_metrics[...,kf_cfg.metric_idxs_to_invert] = 1/(1+perf_metrics[...,kf_cfg.metric_idxs_to_invert])
                raw_metrics_fixed[t_idx, p_idx, i] = perf_metrics
                actual_scaled_metrics_fixed[t_idx, p_idx, i] = metric_scaler.transform(perf_metrics[kf_cfg.metric_idxs].reshape(1, -1)).squeeze()
                # print(f"perf metrics were {raw_metrics_fixed[t_idx, p_idx, i]}")
                # print(f"scaled perf metrics were {metric_scaler.transform(perf_metrics[kf_cfg.metric_idxs].reshape(1, -1)).squeeze()}")
                true_y_scaled = np.dot(cost_weights, metric_scaler.transform(perf_metrics[kf_cfg.metric_idxs].reshape(1, -1)).squeeze())
                actual_costs_fixed[t_idx, p_idx, i] = true_y_scaled


    result_arrays = {"actual_costs": actual_costs_fixed,
                     "raw_metrics": raw_metrics_fixed,
                     "actual_scaled_metrics": actual_scaled_metrics_fixed,
                     "full_traj": full_traj_fixed}
    other_data = {"failed_count": failed_count_ig, "adaptive":experiment_cfg.adaptive}
    return result_arrays, other_data
