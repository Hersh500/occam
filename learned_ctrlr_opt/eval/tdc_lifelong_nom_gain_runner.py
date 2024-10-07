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
from learned_ctrlr_opt.opt.optimize_thru_network import *
from learned_ctrlr_opt.opt.random_search import *
from learned_ctrlr_opt.opt.particle_search import *
from learned_ctrlr_opt.systems.car_controller import CarController, CarControllerParams, pp_track, get_ref_track_pcas_and_scaler, pp_track_curvature
from learned_ctrlr_opt.systems.car_dynamics import CarParams
from learned_ctrlr_opt.systems.robots import TopDownCarRandomStartingState, TopDownCarResetFree
from learned_ctrlr_opt.utils.experiment_utils import *

def eval_success(traj):
    return np.abs(traj[-1][2]) < 15


def tdc_lifelong_nom_gain_runner(experiment_cfg, random_seed):
    kf_checkpoint_dir = experiment_cfg.kf_ckpt_dir
    kf_cfg = OmegaConf.load(os.path.join(kf_checkpoint_dir, "config.yaml"))
    # load scalers
    scalers = get_scalers(kf_cfg.path_to_dataset, kf_cfg.history_length, kf_cfg.metric_idxs,
                          kf_cfg.metric_idxs_to_invert)
    metric_scaler = scalers[2]

    kf_cfg = OmegaConf.load(os.path.join(kf_checkpoint_dir, "config.yaml"))

    test_set = experiment_cfg.test_set_name
    robot_name = experiment_cfg.robot_name
    params_array, seeds, robot_kwargs = load_test_set(robot_name, test_set)
    params_array = [CarParams(*params_array[i]) for i in range(params_array.shape[0])]

    cost_weights = experiment_cfg.cost_weights
    initial_gain = experiment_cfg.initial_gain
    num_trials = experiment_cfg.num_trials

    actual_costs_ig = np.zeros((len(seeds), len(params_array)*num_trials, 1))
    raw_metrics_ig = np.zeros((len(seeds), len(params_array)*num_trials, len(kf_cfg.metric_idxs)))
    actual_scaled_metrics_ig = np.zeros((len(seeds), len(params_array)*num_trials, len(kf_cfg.metric_idxs)))
    failed_count_ig = 0
    for s_idx, seed in enumerate(seeds):
        robot = TopDownCarResetFree(seed=int(seed),
                                    car_params=params_array[0],
                                    gains_to_optimize=kf_cfg.gains_to_optimize,
                                    length=experiment_cfg.segment_length,
                                    max_time=robot_kwargs["max_time"])

        success = False
        # Run ctrlr in a loop
        metrics, traj, env = robot.evaluate_x(initial_gain, render=True)
        success = eval_success(traj)
        if not success:
            print("Could not find an initial gain that worked!")
            break
        np.random.seed(random_seed+s_idx)
        for p_idx, params in enumerate(params_array):
            env.car.set_params(params)
            print(f"seed {random_seed}: on p_idx {p_idx}")
            for i in range(num_trials):
                which_idx = p_idx*num_trials+i
                if not eval_success(traj):
                    print("went out of bounds!")
                    failed_count_ig += 1
                    break

                best_gain_unscaled = initial_gain
                # print(f"---- Step {i} ----")
                # print(f"best_gain_unscaled = {best_gain_unscaled}")
                perf_metrics, traj, env = robot.evaluate_x(best_gain_unscaled, env, render=False)
                perf_metrics[..., kf_cfg.metric_idxs_to_invert] = 1 / (
                            1 + perf_metrics[..., kf_cfg.metric_idxs_to_invert])
                raw_metrics_ig[s_idx, which_idx] = perf_metrics[kf_cfg.metric_idxs]
                # print(f"perf metrics were {perf_metrics}")
                # print(
                #     f"scaled perf metrics were {metric_scaler.transform(perf_metrics[kf_cfg.metric_idxs].reshape(1, -1)).squeeze()}")
                true_y_scaled = np.dot(cost_weights, metric_scaler.transform(
                    perf_metrics[kf_cfg.metric_idxs].reshape(1, -1)).squeeze())

                actual_costs_ig[s_idx, which_idx] = true_y_scaled
                actual_scaled_metrics_ig[s_idx, which_idx] = metric_scaler.transform(
                    perf_metrics[kf_cfg.metric_idxs].reshape(1, -1)).squeeze()

    result_arrays = {"actual_costs": actual_costs_ig,
                     "raw_metrics": raw_metrics_ig,
                     "actual_scaled_metrics": actual_scaled_metrics_ig}
    other_data = {"failed_count": failed_count_ig}
    return result_arrays, other_data
    # save_experiment_data(robot_name, model_name, test_set, result_arrays, other_data=other_data, config=experiment_cfg, num=num)
