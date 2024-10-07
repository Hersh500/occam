import numpy as np
from learned_ctrlr_opt.systems.branin import *
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
from scipy.optimize import minimize, basinhopping

from learned_ctrlr_opt.meta_learning.lsr_net import *
from learned_ctrlr_opt.meta_learning.reptile_net import *
from learned_ctrlr_opt.meta_learning.basis_kf import *
from learned_ctrlr_opt.meta_learning.utils import get_sysid_history_dataset, get_scalers
from learned_ctrlr_opt.opt.random_search import *
from learned_ctrlr_opt.utils.dataset_utils import normalize, denormalize
from learned_ctrlr_opt.systems.branin import BraninFnParamsTrain, BraninFnParamsTest, BraninInputs, Branin
from learned_ctrlr_opt.eval.eval_utils import load_kf_and_scalers, load_reptile_and_scalers
from learned_ctrlr_opt.utils.experiment_utils import load_test_set


def branin_reptile_runner(experiment_cfg, random_seed):
    reptile_checkpoint_dir = experiment_cfg.reptile_ckpt_dir
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    reptile_cfg = OmegaConf.load(os.path.join(reptile_checkpoint_dir, "config.yaml"))
    gain_dim = len(reptile_cfg.gains_to_optimize)

    reptile_net, gain_scaler, history_scaler, metric_scaler = load_reptile_and_scalers(experiment_cfg)

    test_set = experiment_cfg.test_set_name
    robot_name = experiment_cfg.robot_name
    params_array, tasks, robot_kwargs = load_test_set(robot_name, test_set)
    test_params_array = [BraninFnParams(*params_array[i]) for i in range(params_array.shape[0])]

    num_trials = experiment_cfg.num_trials

    reptile_surr_perfs = np.zeros((len(test_params_array), num_trials, 1))
    reptile_perfs = np.zeros((len(test_params_array), num_trials, 1))
    reptile_guesses = np.zeros((len(test_params_array), num_trials, 2))
    expected_scaled_metrics = np.zeros((len(test_params_array), num_trials, 1))
    for p, params in enumerate(test_params_array):
        branin_env = Branin(params, reptile_cfg.gains_to_optimize)
        adapted_net = reptile_net
        np.random.seed(random_seed+p+10000)  # add a fixed offset to differentiate samples b/wn methods
        task_input_data = torch.zeros(num_trials, gain_dim)
        task_target_data = torch.zeros(num_trials, len(reptile_cfg.metric_idxs))
        print(f"on p_idx {p}")
        for trial in range(num_trials):
            def eval_fn_reptile(gain):
                gain_norm = gain_scaler.transform(gain)
                rep_preds = adapted_net(torch.from_numpy(gain_norm).float().to(device))
                rep_preds = metric_scaler.inverse_transform(rep_preds.detach().cpu().numpy()).squeeze()
                return rep_preds, rep_preds

            best_gain, best_acq_cost, best_surr_cost = random_search_simple(experiment_cfg.num_search_samples, eval_fn_reptile, gain_dim,
                                                                            BraninInputs.get_bounds()[reptile_cfg.gains_to_optimize])
            true_value = branin_env.evaluate_x(best_gain)
            reptile_surr_perfs[p, trial] = best_surr_cost
            reptile_perfs[p, trial] = true_value
            reptile_guesses[p, trial] = best_gain
            expected_scaled_metrics[p, trial] = metric_scaler.transform(best_surr_cost.reshape(1, -1)).squeeze()

            best_gain_norm = gain_scaler.transform(best_gain.reshape(1, -1))
            true_value_norm = metric_scaler.transform(true_value.reshape(1, -1))

            task_input_data[trial] = torch.from_numpy(best_gain_norm)
            task_target_data[trial] = torch.from_numpy(true_value_norm)
            adapted_net = adapt_reptile(task_input_data[:trial + 1],
                                        task_target_data[:trial + 1],
                                        reptile_net,
                                        reptile_cfg.inner_lr,
                                        reptile_cfg.num_inner_steps)

    result_arrays = {"expected_costs": reptile_surr_perfs,
                     "actual_costs": reptile_perfs,
                     "tried_gains": reptile_guesses,
                     "expected_scaled_metrics": expected_scaled_metrics}
    other_data = {}
    return result_arrays, other_data
