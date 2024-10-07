import numpy as np
from learned_ctrlr_opt.systems.branin import *
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
from learned_ctrlr_opt.eval.eval_utils import load_kf_and_scalers
from learned_ctrlr_opt.utils.experiment_utils import load_test_set


def branin_kf_runner(experiment_cfg, random_seed, adapt=True):
    kf_checkpoint_dir = experiment_cfg.kf_ckpt_dir
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    kf_cfg = OmegaConf.load(os.path.join(kf_checkpoint_dir, "config.yaml"))
    gain_dim = len(kf_cfg.gains_to_optimize)

    kf_network, gain_scaler, history_scaler, metric_scaler = load_kf_and_scalers(experiment_cfg)

    test_set = experiment_cfg.test_set_name
    robot_name = experiment_cfg.robot_name
    params_array, tasks, robot_kwargs = load_test_set(robot_name, test_set)
    test_params_array = [BraninFnParams(*params_array[i]) for i in range(params_array.shape[0])]

    num_trials = experiment_cfg.num_trials

    kf_surr_perfs = np.zeros((len(test_params_array), num_trials, 1))
    kf_perfs = np.zeros((len(test_params_array), num_trials, 1))
    kf_guesses = np.zeros((len(test_params_array), num_trials, 2))
    expected_scaled_metrics = np.zeros((len(test_params_array), num_trials, 1))
    expected_variances = np.zeros((len(test_params_array), num_trials, 1))
    for p, params in enumerate(test_params_array):
        branin_env = Branin(params, kf_cfg.gains_to_optimize)

        # Don't use meta-learned hyperparameters
        if experiment_cfg.no_meta:
            kf_network.initialize_priors()

        weights = kf_network.last_layer_prior
        sigma = torch.mm(kf_network.last_layer_prior_cov_sqrt, torch.t(kf_network.last_layer_prior_cov_sqrt)).to(device)
        Q = torch.mm(kf_network.Q_sqrt, torch.t(kf_network.Q_sqrt)).to(device)
        R = torch.mm(kf_network.R_sqrt, torch.t(kf_network.R_sqrt)).to(device)

        np.random.seed(random_seed+p)
        sigma_weight = experiment_cfg.kf_sigma_penalty
        print(f"On params idx {p}")
        for trial in range(num_trials):
            def eval_fn_kf(gain):
                gain_norm = gain_scaler.transform(gain)
                kf_preds, kf_sigma = last_layer_prediction_uncertainty_aware(torch.from_numpy(gain_norm).to(device),
                                                                             kf_network,
                                                                             weights,
                                                                             sigma)
                kf_preds = metric_scaler.inverse_transform(kf_preds.detach().cpu().numpy()).squeeze()
                kf_sigma = np.sqrt(kf_sigma.cpu().detach().numpy().squeeze() * metric_scaler.scale_ ** 2).squeeze()
                return kf_preds + sigma_weight * kf_sigma, kf_preds

            best_gain, best_acq_cost, best_surr_cost = random_search_simple(experiment_cfg.num_search_samples,
                                                                            eval_fn_kf,
                                                                            gain_dim,
                                                                            BraninInputs.get_bounds()[kf_cfg.gains_to_optimize])
            true_value = branin_env.evaluate_x(best_gain)
            kf_surr_perfs[p, trial] = best_surr_cost
            kf_perfs[p, trial] = true_value
            kf_guesses[p, trial] = best_gain

            input_test = torch.from_numpy(best_gain).unsqueeze(0).float().to(device)
            best_y_mean, best_y_sigma = last_layer_prediction_uncertainty_aware(input_test,
                                                                                kf_network,
                                                                                weights,
                                                                                sigma)
            expected_scaled_metrics[p, trial] = best_y_mean.cpu().detach().numpy()
            expected_variances[p, trial] = best_y_sigma.cpu().detach().numpy()

            best_gain_norm = gain_scaler.transform(best_gain.reshape(1, -1))
            true_value_norm = metric_scaler.transform(true_value.reshape(1, -1))
            # fit network - boilerplate that needs to be put in method
            phi = kf_network(torch.from_numpy(best_gain_norm).float().to(device)).squeeze(0)
            target = torch.from_numpy(true_value_norm).float().to(device)
            if adapt:
                for _ in range(experiment_cfg.num_replay_steps+1):
                    weights, sigma, K = kalman_step(weights, sigma, target, phi, Q, R)

    result_arrays = {"expected_costs": kf_surr_perfs,
                     "actual_costs": kf_perfs,
                     "tried_gains": kf_guesses,
                     "expected_scaled_metrics": expected_scaled_metrics,
                     "expected_variances": expected_variances}
    other_data = {}
    return result_arrays, other_data
