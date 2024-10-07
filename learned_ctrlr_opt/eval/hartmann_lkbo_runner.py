import numpy as np
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

from learned_ctrlr_opt.opt.random_search import *
from learned_ctrlr_opt.systems.hartmann import HartmannFnParams, HartmannInputs, Hartmann
from learned_ctrlr_opt.meta_learning.lkbo import DeepGPRegressionModel, train_dk_gp, get_covar_module
from learned_ctrlr_opt.eval.eval_utils import load_kf_and_scalers
from learned_ctrlr_opt.utils.experiment_utils import load_test_set


def hartmann_lkbo_runner(experiment_cfg, random_seed, adapt=True):
    kf_checkpoint_dir = experiment_cfg.kf_ckpt_dir
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    kf_cfg = OmegaConf.load(os.path.join(kf_checkpoint_dir, "config.yaml"))
    gain_dim = len(kf_cfg.gains_to_optimize)

    kf_network, gain_scaler, history_scaler, metric_scaler = load_kf_and_scalers(experiment_cfg)

    test_set = experiment_cfg.test_set_name
    robot_name = experiment_cfg.robot_name
    params_array, tasks, robot_kwargs = load_test_set(robot_name, test_set)
    test_params_array = [HartmannFnParams(*params_array[i]) for i in range(params_array.shape[0])]

    num_trials = experiment_cfg.num_trials

    lkbo_surr_perfs = np.zeros((len(test_params_array), num_trials, 1))
    lkbo_perfs = np.zeros((len(test_params_array), num_trials, 1))
    lkbo_guesses = np.zeros((len(test_params_array), num_trials, len(kf_cfg.gains_to_optimize)))
    # expected_scaled_metrics = np.zeros((len(test_params_array), num_trials, 1))
    expected_variances = np.zeros((len(test_params_array), num_trials, 1))
    for p, params in enumerate(test_params_array):
        env = Hartmann(params, kf_cfg.gains_to_optimize)

        kf_network.use_last_layer = True
        np.random.seed(random_seed+p)
        sigma_weight = experiment_cfg.bo_sigma_penalty
        print(f"On params idx {p}")
        # very simple data for this one
        train_x = torch.zeros((num_trials, len(kf_cfg.gains_to_optimize))).float().to(device)
        train_y = torch.zeros((num_trials, 1)).float().to(device)
        for trial in range(num_trials):
            def eval_fn_kf(gain):
                gain_norm = torch.from_numpy(gain_scaler.transform(gain)).float().to(device)
                kf_preds = metric_scaler.inverse_transform(kf_network(gain_norm).cpu().detach().numpy())
                return kf_preds, kf_preds

            def eval_fn_gp(gain):
                gain_norm = torch.from_numpy(gain_scaler.transform(gain)).float().to(device)
                distr = gp_model(gain_norm)
                # what y does the model output here...? Scaled or unscaled variances?
                mean = distr.mean.cpu().detach().numpy()
                variance = distr.variance.cpu().detach().numpy()
                return mean + sigma_weight*variance, mean

            if trial <= experiment_cfg.lkbo_num_explore_iters:
                best_gain, best_acq_cost, best_surr_cost = random_search_simple(experiment_cfg.num_search_samples,
                                                                                eval_fn_kf,
                                                                                gain_dim,
                                                                                HartmannInputs.get_bounds()[kf_cfg.gains_to_optimize])
            else:
                best_gain, best_acq_cot, best_surr_cost = random_search_simple(experiment_cfg.num_search_samples,
                                                                               eval_fn_gp,
                                                                               gain_dim,
                                                                               HartmannInputs.get_bounds()[kf_cfg.gains_to_optimize])
            true_value = env.evaluate_x(best_gain)
            lkbo_surr_perfs[p, trial] = best_surr_cost
            lkbo_perfs[p, trial] = true_value
            lkbo_guesses[p, trial] = best_gain

            input_test = torch.from_numpy(gain_scaler.transform(best_gain.reshape((1, -1)))).float().to(device)

            if trial > experiment_cfg.lkbo_num_explore_iters:
                var = gp_model(input_test).variance
            else:
                var = get_covar_module(6)(input_test)
            expected_variances[p, trial] = var.cpu().detach().numpy()
            true_value_norm = metric_scaler.transform(true_value.reshape(1, -1))
            # target = torch.from_numpy(true_value_norm).float().to(device)
            # target = torch.from_numpy(true_value).unsqueeze(0).float().to(device)
            train_x[trial] = input_test
            train_y[trial] = true_value
            gp_model, mll = train_dk_gp(train_x[:trial+1],
                                        train_y[:trial+1].squeeze(),
                                        kf_network,
                                        experiment_cfg.lkbo_num_train_iters,
                                        len(kf_cfg.metric_idxs),
                                        device)
            gp_model.eval()

    result_arrays = {"expected_costs": lkbo_surr_perfs,
                     "actual_costs": lkbo_perfs,
                     "tried_gains": lkbo_guesses,
                     "expected_variances": expected_variances}
    other_data = {}
    return result_arrays, other_data
