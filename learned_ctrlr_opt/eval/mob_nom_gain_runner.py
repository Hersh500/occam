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

from learned_ctrlr_opt.meta_learning.lsr_net import *
from learned_ctrlr_opt.meta_learning.basis_kf import *
from learned_ctrlr_opt.meta_learning.utils import get_sysid_history_dataset, get_scalers
from learned_ctrlr_opt.opt.optimize_thru_network import *
from learned_ctrlr_opt.opt.random_search import *
from learned_ctrlr_opt.utils.experiment_utils import *
from learned_ctrlr_opt.eval.eval_utils import *
from learned_ctrlr_opt.utils.dataset_utils import normalize
import time




def mob_nom_gain_runner(experiment_cfg, random_seeds):
    kf_checkpoint_dir = experiment_cfg.kf_ckpt_dir
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    kf_cfg = OmegaConf.load(os.path.join(kf_checkpoint_dir, "config.yaml"))
    gain_dim = len(kf_cfg.gains_to_optimize)

    kf_network, gain_scaler, history_scaler, metric_scaler = load_kf_and_scalers(experiment_cfg)

    test_set = experiment_cfg.test_set_name
    robot_name = experiment_cfg.robot_name
    params_array, tasks, robot_kwargs = load_test_set(robot_name, test_set)
    # params_array = [Go1HiddenParams(*params_array[i]) for i in range(params_array.shape[0])]
    cmds = tasks

    traj_len = robot_kwargs["eval_length_s"] * 50
    cost_weights = np.array(experiment_cfg.cost_weights)
    num_trials = experiment_cfg.num_trials

    assert num_trials == cmds.shape[1]

    expected_costs = np.zeros((len(random_seeds), len(cmds), len(params_array), num_trials, 1))
    actual_costs = np.zeros((len(random_seeds), len(cmds), len(params_array), num_trials, 1))
    raw_metrics = np.zeros((len(random_seeds), len(cmds), len(params_array), num_trials, len(kf_cfg.metric_idxs)))
    full_traj = np.zeros((len(random_seeds), len(cmds), len(params_array), traj_len * num_trials, kf_cfg.traj_dim))
    tried_gains = np.zeros((len(random_seeds), len(cmds), len(params_array), num_trials, gain_dim))
    all_weights = np.zeros((len(random_seeds), len(cmds), len(params_array), num_trials, kf_cfg.n_bases))
    actual_scaled_metrics = np.zeros((len(random_seeds), len(cmds), len(params_array), num_trials, len(kf_cfg.metric_idxs)))
    failed_counts = np.zeros(len(random_seeds))
    n_envs = params_array.shape[0]
    print(f"Creating {n_envs} parallel envs")
    robot = MoBLocomotion_ResetFree(gains_to_optimize=kf_cfg.gains_to_optimize,
                                    params=params_array,
                                    num_envs=n_envs,
                                    render=False,
                                    eval_length_s=robot_kwargs["eval_length_s"])

    initial_gains = np.array([experiment_cfg.initial_gain for w in range(n_envs)])
    for s_idx, seed in enumerate(random_seeds):
        print(f"---On seed {s_idx}---")
        for t_idx, cmd in enumerate(cmds):
            print(f"On t_idx {t_idx}")
            np.random.seed(seed + t_idx)
            torch.manual_seed(seed + t_idx)
            robot.reset_env()

            cmd_rpt = np.zeros((n_envs, 3))
            cmd_rpt[:,:] = cmds[t_idx,0,:]
            metrics, obtained_trajs, successes = robot.evaluate_x(initial_gains, cmd=cmd_rpt)
            success_mask = np.array(successes, dtype=bool)
            for i in range(num_trials):
                for p_idx in range(params_array.shape[0]):
                    if not successes[p_idx]:
                        if success_mask[p_idx]:
                            failed_counts[s_idx] += 1
                            success_mask[p_idx] = False
                        continue

                    traj_lim = obtained_trajs[t_idx, -kf_cfg.history_length:]
                    traj_lim = np.expand_dims(traj_lim, 0)
                    full_traj[s_idx, t_idx, p_idx, i*traj_len:(i+1)*traj_len] = obtained_trajs[p_idx]
                tried_gains[s_idx,t_idx, :,i] = initial_gains
                # Once all gains are obtained, run the simulation
                cmd_rpt = np.zeros((n_envs, 3))
                cmd_rpt[:,:] = cmds[t_idx,i,:]
                perf_metrics, obtained_trajs, successes = robot.evaluate_x(initial_gains, cmd_rpt)
                perf_metrics[...,kf_cfg.metric_idxs_to_invert] = 1/(1+perf_metrics[...,kf_cfg.metric_idxs_to_invert])
                perf_metrics_scaled = metric_scaler.transform(perf_metrics[...,kf_cfg.metric_idxs])
                true_y_scaled = np.dot(perf_metrics_scaled, cost_weights)  # is this the right shape?
                actual_costs[s_idx, t_idx, success_mask, i] = true_y_scaled.reshape((-1, 1))[success_mask]

                raw_metrics[s_idx,t_idx,success_mask,i] = perf_metrics[success_mask][:,kf_cfg.metric_idxs]
                actual_scaled_metrics[s_idx,t_idx,success_mask,i] = perf_metrics_scaled[success_mask]

    results = []
    for s_idx, seed in enumerate(random_seeds):
        result = ({"expected_costs": expected_costs[s_idx],
                   "actual_costs": actual_costs[s_idx],
                   "raw_metrics": raw_metrics[s_idx],
                   "tried_gains": tried_gains[s_idx],
                   "all_weights": all_weights[s_idx],
                   "actual_scaled_metrics": actual_scaled_metrics[s_idx],
                   "full_traj": full_traj[s_idx]})
        results.append((result, {"failed_count":failed_counts[s_idx]}))
    return results

def mob_nom_gain_runner_idx(experiment_cfg, random_seed, cmd_idx, param_idx):
    kf_checkpoint_dir = experiment_cfg.kf_ckpt_dir
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    kf_cfg = OmegaConf.load(os.path.join(kf_checkpoint_dir, "config.yaml"))
    gain_dim = len(kf_cfg.gains_to_optimize)

    kf_network, gain_scaler, history_scaler, metric_scaler = load_kf_and_scalers(experiment_cfg)

    test_set = experiment_cfg.test_set_name
    robot_name = experiment_cfg.robot_name
    params_array, tasks, robot_kwargs = load_test_set(robot_name, test_set)
    # params_array = [Go1HiddenParams(*params_array[i]) for i in range(params_array.shape[0])]
    cmds = tasks

    params_array = params_array[param_idx:param_idx+1]
    print(f"Params are, {params_array}")
    t_idx = cmd_idx

    traj_len = robot_kwargs["eval_length_s"] * 50
    cost_weights = np.array(experiment_cfg.cost_weights)
    num_trials = experiment_cfg.num_trials

    assert num_trials <= cmds.shape[1]

    n_envs = 1
    np.random.seed(experiment_cfg.seeds[random_seed] + t_idx)
    torch.manual_seed(experiment_cfg.seeds[random_seed] + t_idx)
    robot = MoBLocomotion_ResetFree(gains_to_optimize=kf_cfg.gains_to_optimize,
                                    params=params_array,
                                    num_envs=n_envs,
                                    render=True,
                                    eval_length_s=robot_kwargs["eval_length_s"])
    time.sleep(3)

    initial_gains = np.array([experiment_cfg.initial_gain for w in range(n_envs)])
    robot.reset_env()

    cmd_rpt = np.zeros((n_envs, 3))
    cmd_rpt[:,:] = cmds[t_idx,0,:]
    metrics, obtained_trajs, successes = robot.evaluate_x(initial_gains, cmd=cmd_rpt)
    success_mask = np.array(successes, dtype=bool)
    p_idx = 0
    for i in range(num_trials):
        print(f"On Trial {i}")
        if not successes[p_idx]:
            print("Crashed!")
            time.sleep(4)
            continue

        # Once all gains are obtained, run the simulation
        cmd_rpt = np.zeros((n_envs, 3))
        cmd_rpt[:,:] = cmds[t_idx,i,:]
        perf_metrics, obtained_trajs, successes = robot.evaluate_x(initial_gains, cmd_rpt)
        perf_metrics[...,kf_cfg.metric_idxs_to_invert] = 1/(1+perf_metrics[...,kf_cfg.metric_idxs_to_invert])
        perf_metrics_scaled = metric_scaler.transform(perf_metrics[...,kf_cfg.metric_idxs])
        true_y_scaled = np.dot(perf_metrics_scaled, cost_weights)  # is this the right shape?
