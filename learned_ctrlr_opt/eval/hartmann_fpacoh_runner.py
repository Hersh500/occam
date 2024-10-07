import numpy as np
from learned_ctrlr_opt.systems.hartmann import *
import h5py
import os
from omegaconf import OmegaConf
from meta_bo.models import FPACOH_MAP_GP

from learned_ctrlr_opt.opt.random_search import *
from learned_ctrlr_opt.utils.experiment_utils import load_test_set

def hartmann_fpacoh_runner(fpacoh_model, experiment_cfg, random_seed):
    kf_checkpoint_dir = experiment_cfg.kf_ckpt_dir
    kf_cfg = OmegaConf.load(os.path.join(kf_checkpoint_dir, "config.yaml"))

    test_set = experiment_cfg.test_set_name
    robot_name = experiment_cfg.robot_name
    params_array, tasks, robot_kwargs = load_test_set(robot_name, test_set)
    test_params_array = [HartmannFnParams(*params_array[i]) for i in range(params_array.shape[0])]

    num_trials = experiment_cfg.num_trials

    expected_variances = np.zeros((len(test_params_array), num_trials, 1))
    fpacoh_surr_perfs = np.zeros((len(test_params_array), num_trials, 1))
    fpacoh_perfs = np.zeros((len(test_params_array), num_trials, 1))
    fpacoh_guesses = np.zeros((len(test_params_array), num_trials, len(kf_cfg.gains_to_optimize)))

    def eval_fn_pacoh(gain):
        pacoh_pred, pacoh_std = fpacoh_model.predict(gain)
        return pacoh_pred + experiment_cfg.pacoh_sigma_penalty * pacoh_std, pacoh_pred

    for p in range(len(test_params_array)):
        params_array = test_params_array[p]
        env = Hartmann(params_array, kf_cfg.gains_to_optimize)
        fpacoh_model.reset_to_prior()
        np.random.seed(random_seed+p+20000)
        print(f"On p_idx {p}")
        for trial in range(num_trials):
            best_gain, best_acq_cost, best_surr_cost = random_search_simple(experiment_cfg.num_search_samples,
                                                                            eval_fn_pacoh,
                                                                            len(kf_cfg.gains_to_optimize),
                                                                            HartmannInputs.get_bounds()[kf_cfg.gains_to_optimize])
            true_value = env.evaluate_x(best_gain)
            fpacoh_surr_perfs[p, trial] = best_surr_cost
            fpacoh_perfs[p, trial] = true_value
            fpacoh_guesses[p, trial] = best_gain
            # FOR SOME REASON THIS DOESN"T WORK?
            # pacoh_pred, pacoh_std = fpacoh_model.predict(best_gain.reshape((1, -1)))
            # expected_variances[p, trial] = pacoh_std**2  # saving variance, not std
            fpacoh_model.add_data(best_gain.reshape(1, -1), true_value)

    result_arrays = {"expected_costs": fpacoh_surr_perfs,
                     "actual_costs": fpacoh_perfs,
                     "tried_gains": fpacoh_guesses,
                     "expected_variances": expected_variances}
    other_data = {}
    return result_arrays, other_data
