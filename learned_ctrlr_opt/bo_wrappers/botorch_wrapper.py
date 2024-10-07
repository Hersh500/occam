import wandb
import os
import pickle

from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement, qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.sampling import SobolQMCNormalSampler

from learned_ctrlr_opt.bo_wrappers.botorch_validation import test_model_fit, validate_model
from learned_ctrlr_opt.bo_wrappers.tuning_config import BayesOptParams
from learned_ctrlr_opt.bo_wrappers.botorch_utils import *
from learned_ctrlr_opt.bo_wrappers.sys_id import *
from learned_ctrlr_opt.bo_wrappers.botorch_models import MixedModel, \
    RGPEMixedModel, NNSlicePrior, prepare_and_train_prior_nn
from learned_ctrlr_opt.utils.dataset_utils import normalize, denormalize


def make_acqf(opt_config: BayesOptParams, model, cur_best_obj, transform, X):
    if opt_config.acq_fn.lower() == "ucb":
        return UpperConfidenceBound(model, opt_config.beta, posterior_transform=transform)
    elif opt_config.acq_fn.lower() == "ei":
        return ExpectedImprovement(model, best_f=cur_best_obj, posterior_transform=transform)
    elif opt_config.acq_fn.lower() == "nei":
        sampler = SobolQMCNormalSampler(256)
        return qNoisyExpectedImprovement(model, X_baseline=X, sampler=sampler,
                                         posterior_transform=transform)
    else:
        raise NotImplementedError(f"acquisition function {opt_config.acq_fn} is not supported.")




def botorch_optimize_simple(robot: Robot):
    obj_values = []
    best_obj_values = []
    acq_values = []
    best_gain = []
    test_errors = []
    test_variances = []

    cur_best_obj = -np.inf
    cur_best_obj_norm = -100
    num_gains = len(robot.gains_to_optimize)



def botorch_optimize(robot: Robot,
                     tuning_config: ExperimentConfig,
                     wandb_run=None,
                     x_test: np.ndarray = None,
                     y_test: np.ndarray = None):
    obj_values = []
    best_obj_values = []
    acq_values = []
    best_gain = []
    test_errors = []
    test_variances = []

    cur_best_obj = -np.inf
    cur_best_obj_norm = -100
    opt_params = tuning_config.opt_params
    num_evals = opt_params.num_evals
    num_gains = len(robot.gains_to_optimize)
    use_ground_truth_theta = tuning_config.use_ground_truth_theta
    if not use_ground_truth_theta:
        params_to_slice = robot.gains_to_optimize
        theta_idxs = np.array([], dtype=int)
    else:
        if not opt_params.use_priors:
            raise NotImplementedError(
                "Using ground truth thetas without priors is not supported, as this behavior is undefined.")
        else:
            print("Using ground truth theta in bayesian optimization")
        theta_idxs = np.array(tuning_config.robot_info.thetas_to_sweep, dtype=int)
        ground_truth_theta = robot.get_thetas()[theta_idxs]
        print(f"ground truth thetas are {ground_truth_theta}")
        num_thetas = len(tuning_config.robot_info.thetas_to_sweep)
        params_to_slice = np.append(robot.gains_to_optimize,
                                    np.array(theta_idxs, dtype=int) + num_gains)

    num_points = opt_params.n_initial_points
    num_priors = opt_params.num_priors_to_use
    if not opt_params.use_priors:
        x0, y0 = generate_data(num_points, robot)
    else:
        x0 = np.load(os.path.join(robot.prior_dir, tuning_config.prior_x0_file))[:num_priors]
        y0 = np.load(os.path.join(robot.prior_dir, tuning_config.prior_y0_file))[:num_priors]
    with open(os.path.join(robot.prior_dir, tuning_config.scaler_file), 'rb') as f:
        y_scaler = pickle.load(f)
    prior_model, prior_mll, x0_torch, y0_torch, y_scaler, all_bounds, device = prepare_and_train_prior(x0, y0, robot,
                                                                                                       theta_idxs,
                                                                                                       y_scaler=y_scaler,
                                                                                                       cpu=True)
    if not opt_params.use_priors and wandb_run is not None:
        for i in range(num_points):
            obj = np.dot(y_scaler.transform(y0[i:i+1]), tuning_config.weights)
            if obj > cur_best_obj_norm:
                cur_best_obj = np.dot(y0[i], tuning_config.weights)
                cur_best_obj_norm = obj
                best_gain = normalize(x0[i], robot.get_gain_bounds())
            wandb_run.log({"Evaluation/y_eval": obj,
                           "Evaluation/best_y_eval": cur_best_obj_norm})

    if use_ground_truth_theta:
        ground_truth_theta_norm = normalize(ground_truth_theta, all_bounds[num_gains:])

    all_bounds_norm = torch.from_numpy(np.array([[0, 1] for j in range(x0_torch.size(2))], dtype=float).T)

    x = x0_torch.to(device)
    y = y0_torch.to(device)
    model = prior_model.to(device)
    transform = ScalarizedPosteriorTransform(weights=torch.from_numpy(np.array(tuning_config.weights)))

    x_eval = np.zeros((num_evals, num_gains))
    y_eval = np.zeros((num_evals, len(robot.perf_metric_names())))
    perf_metric_names = robot.perf_metric_names()
    for i in range(num_evals):
        mll, model = initialize_model(x, y, state_dict=model.state_dict())
        fit_gpytorch_mll(mll)
        step_log_dict = {}
        if False and x_test is not None and y_test is not None:
            ce_model_mae, model_avg_variance = validate_model(model,
                                                              x_test,
                                                              y_test,
                                                              all_bounds,
                                                              y_scaler,
                                                              device)
            test_errors.append(ce_model_mae)
            test_variances.append(model_avg_variance)
            for j in range(len(perf_metric_names)):
                step_log_dict["Validation/Test_Err " + perf_metric_names[j]] = ce_model_mae[j]
                step_log_dict["Validation/Test_Var " + perf_metric_names[j]] = model_avg_variance[j]
            # wandb.log({"val_err": ce_model_mae, "avg_val_var": model_avg_variance})
        # acqf = UpperConfidenceBound(model_ucb, beta=1.0, posterior_transform=transform)
        # acqf = ExpectedImprovement(model_ucb, best_f=cur_best_obj, posterior_transform=transform)
        acqf = make_acqf(tuning_config.opt_params, model, cur_best_obj_norm, transform, x)
        if not use_ground_truth_theta:
            candidate, acq_value = optimize_acqf(acqf, bounds=all_bounds_norm, q=1, num_restarts=5, raw_samples=30)
        else:
            fixed_thetas = {i: ground_truth_theta_norm[i - num_gains] for i in range(num_gains, num_gains + num_thetas)}
            candidate, acq_value = optimize_acqf(acqf, bounds=all_bounds_norm, q=1, num_restarts=5, raw_samples=30,
                                                 fixed_features=fixed_thetas)
        candidate_np = candidate.cpu().detach().numpy()[0][:num_gains]
        x_eval[i] = candidate_np
        obj = robot.evaluate_x(denormalize(candidate_np, all_bounds[:num_gains]))
        # y_eval[i] = y_scaler.transform(obj.reshape(1, obj.shape[0]))
        y_eval[i] = obj  # don't scale this data because the scaler is inconsistent across different optimization runs
        obj_norm = y_scaler.transform(obj.reshape(1, obj.shape[0]))
        if np.dot(obj_norm, tuning_config.weights) > cur_best_obj_norm:
            cur_best_obj = np.dot(y_eval[i], tuning_config.weights)
            cur_best_obj_norm = np.dot(obj_norm, tuning_config.weights)
            best_gain = x_eval[i]
        best_obj_values.append(cur_best_obj_norm)
        obj_values.append(obj_norm)
        acq_values.append(acq_value)
        diffs = test_model_fit(model, candidate_np.reshape(1, -1), obj.reshape(1, -1),
                               y_scaler, device, x_already_normalized=True)
        for j, name in enumerate(perf_metric_names):
            step_log_dict["Validation/Model Residual on "+name] = diffs[j]
        if use_ground_truth_theta:
            x = build_all_x(x0[:, params_to_slice], ground_truth_theta, all_bounds, eval_gains=x_eval[:i + 1, :]).to(
                device)
        else:
            x = build_all_x(x0[:, params_to_slice], np.array([]), all_bounds, eval_gains=x_eval[:i + 1, :]).to(device)
        y_cur = y_scaler.transform(obj.reshape(1, obj.shape[0]))
        # Now scale the data here, for training the GP
        y = torch.cat([y, torch.from_numpy(y_cur).view(1, 1, len(robot.perf_metric_names())).to(device)], dim=1)
        print(
            f"on iteration {i}, the observed value was {np.dot(obj_norm, tuning_config.weights)}.....best is now {cur_best_obj_norm}")
        step_log_dict["Evaluation/y_eval"] = np.dot(obj_norm, tuning_config.weights)
        step_log_dict["Evaluation/best_y_eval"] = cur_best_obj_norm
        if wandb_run is not None:
            wandb_run.log(step_log_dict)
    if wandb_run is not None:
        wandb_run.log({"Evaluated Gains": wandb.Table(columns=robot.get_gain_names()[robot.gains_to_optimize].tolist(),
                              data=denormalize(x_eval, all_bounds[:num_gains]).tolist())})
    result = Result(best_gain,
                    cur_best_obj,
                    obj_values,
                    x0,
                    y0,
                    x_eval,
                    y_eval,
                    model.state_dict(),
                    robot,
                    None,
                    np.array(test_errors),
                    np.array(test_variances))
    # wandb.log({"result_obj": result})
    return result


def random_search(robot: Robot,
                  tuning_config: ExperimentConfig,
                  wandb_run=None):
    obj_values = []
    best_obj_values = []
    best_gain = []

    cur_best_obj = -np.inf
    cur_best_obj_norm = -100
    opt_params = tuning_config.opt_params
    num_evals = opt_params.num_evals
    num_gains = len(robot.gains_to_optimize)

    x_eval = np.zeros((num_evals, num_gains))
    y_eval = np.zeros((num_evals, len(robot.perf_metric_names())))
    with open(os.path.join(robot.prior_dir, tuning_config.scaler_file), 'rb') as f:
        y_scaler = pickle.load(f)
    for i in range(num_evals):
        step_log_dict = {}
        candidate_np = np.array(robot.ControllerParamsT.generate_random(robot.gains_to_optimize).get_list())
        x_eval[i] = candidate_np[robot.gains_to_optimize]
        obj = robot.evaluate_x(x_eval[i])
        y_eval[i] = obj
        obj_norm = y_scaler.transform(obj.reshape(1, obj.shape[0]))
        if np.dot(obj_norm, tuning_config.weights) > cur_best_obj_norm:
            cur_best_obj = np.dot(y_eval[i], tuning_config.weights)
            cur_best_obj_norm = np.dot(obj_norm, tuning_config.weights)
            best_gain = x_eval[i]
        best_obj_values.append(cur_best_obj_norm)
        obj_values.append(obj_norm)
        print(
            f"on iteration {i}, the observed value was {np.dot(obj_norm, tuning_config.weights)}.....best is now {cur_best_obj_norm}")
        step_log_dict["Evaluation/y_eval"] = np.dot(obj_norm, tuning_config.weights)
        step_log_dict["Evaluation/best_y_eval"] = cur_best_obj_norm
        if wandb_run is not None:
            wandb_run.log(step_log_dict)
    if wandb_run is not None:
        wandb_run.log({"Evaluated Gains": wandb.Table(columns=robot.get_gain_names()[robot.gains_to_optimize].tolist(),
                                                      data=x_eval)})
    result = Result(best_gain,
                    cur_best_obj,
                    obj_values,
                    np.array([]),
                    np.array([]),
                    x_eval,
                    y_eval,
                    {},
                    robot,
                    None,
                    np.array([]),
                    np.array([]))
    return result


# def botorch_optimize_particles(robot: Robot,
#                                tuning_config: ExperimentConfig,
#                                x_test=None,
#                                y_test=None,
#                                wandb_run=None):
#     # For logging purposes.
#     obj_values = []
#     best_obj_values = []
#     acq_values = []
#     best_gain = []
#     particle_history = []
#     weight_history = []
#     test_errors = []
#     test_variances = []
#
#     cur_best_obj = -np.inf
#     cur_best_obj_norm = -100
#     opt_params = tuning_config.opt_params
#     num_evals = opt_params.num_evals
#     num_gains = len(robot.gains_to_optimize)
#     theta_idxs = np.array(tuning_config.robot_info.thetas_to_sweep, dtype=int)
#     num_thetas = len(theta_idxs)
#     num_priors = opt_params.num_priors_to_use
#     params_to_slice = np.append(robot.gains_to_optimize,
#                                 np.array(theta_idxs, dtype=int) + len(robot.gains_to_optimize))
#     resample_every = tuning_config.sys_id.resample_every
#     explore_noise = tuning_config.sys_id.explore_noise
#
#     x0 = np.load(os.path.join(robot.prior_dir, tuning_config.prior_x0_file))[:num_priors]
#     y0 = np.load(os.path.join(robot.prior_dir, tuning_config.prior_y0_file))[:num_priors]
#     with open(os.path.join(robot.prior_dir, tuning_config.scaler_file), 'rb') as f:
#         y_scaler = pickle.load(f)
#     prior_model, prior_mll, x0_torch, y0_torch, y_scaler, all_bounds_wt, device = prepare_and_train_prior(x0, y0, robot,
#                                                                                                           theta_idxs,
#                                                                                                           y_scaler=y_scaler)
#     gain_bounds = all_bounds_wt[:num_gains, :]
#     x0_torch = build_all_x(x0[:, robot.gains_to_optimize], [], gain_bounds)
#
#     # all_bounds_torch = torch.from_numpy(gain_bounds.T)
#     gain_bounds_norm = torch.from_numpy(np.array([[0, 1] for j in range(x0_torch.size(2))], dtype=float).T)
#
#     x = x0_torch
#     y = y0_torch
#     transform = ScalarizedPosteriorTransform(weights=torch.from_numpy(np.array(tuning_config.weights)))
#
#     x_eval = np.zeros((num_evals, len(robot.gains_to_optimize)))
#     y_eval = np.zeros((num_evals, len(robot.perf_metric_names())))
#     y_eval_norm = np.zeros((num_evals, len(robot.perf_metric_names())))
#
#     # Particle filter stuff
#     weights = np.ones(x0.shape[0]) * 1 / x0.shape[0]
#     theta_particles_orig = x0[:, num_gains + theta_idxs]
#     theta_particles_norm = normalize(x0[:, num_gains + theta_idxs], all_bounds_wt[num_gains:])
#     # if there are more desired particles than priors, we need to resample to add some
#     theta_particles_norm, weights = resample_particles_simple(theta_particles_norm, weights,
#                                                               num=tuning_config.sys_id.num_particles)
#     # add noise to remove initial duplicates
#     theta_particles_norm = predict(theta_particles_norm, tuning_config.sys_id.explore_noise)
#     theta_particles_torch = torch.unsqueeze(torch.from_numpy(theta_particles_norm), 0).double()
#     model = WeightedSliceGP(prior_model, weights=torch.from_numpy(weights),
#                             theta_particles=theta_particles_torch, x_dim=num_gains, device=device)
#
#     print(f"true thetas are {robot.get_thetas()}")
#     perf_metric_names = robot.perf_metric_names()
#     for i in range(num_evals):
#         # acqf = UpperConfidenceBound(model_ucb, beta=0.3, posterior_transform=transform)
#         # acqf = ExpectedImprovement(model_ucb, best_f=cur_best_obj, posterior_transform=transform)
#         step_log_dict = {}
#         if x_test is not None and y_test is not None:
#             ce_model_mae, model_avg_variance = validate_model(model,
#                                                               x_test,
#                                                               y_test,
#                                                               gain_bounds,
#                                                               y_scaler,
#                                                               device)
#             test_errors.append(ce_model_mae)
#             test_variances.append(model_avg_variance)
#             for j in range(len(perf_metric_names)):
#                 step_log_dict["Validation/Test_Err " + perf_metric_names[j]] = ce_model_mae[j]
#                 step_log_dict["Validation/Test_Var " + perf_metric_names[j]] = model_avg_variance[j]
#         acqf = make_acqf(tuning_config.opt_params, model, cur_best_obj_norm, transform)
#         candidate, acq_value = optimize_acqf(acqf, bounds=gain_bounds_norm, q=1, num_restarts=5, raw_samples=30)
#         candidate_np = candidate.cpu().detach().numpy()[0]
#         x_eval[i] = candidate_np
#         obj = robot.evaluate_x(denormalize(candidate_np, gain_bounds))
#         y_eval[i] = obj
#         obj_norm = y_scaler.transform(obj.reshape(1, -1))
#         y_eval_norm[i] = obj_norm
#         if np.dot(obj_norm, tuning_config.weights) > cur_best_obj_norm:
#             cur_best_obj = np.dot(y_eval[i], tuning_config.weights)
#             cur_best_obj_norm = np.dot(obj_norm, tuning_config.weights)
#             best_gain = x_eval[i]
#         best_obj_values.append(cur_best_obj_norm)
#         obj_values.append(obj_norm)
#         acq_values.append(acq_value)
#         particle_history.append(denormalize(theta_particles_norm, all_bounds_wt[num_gains:]))
#         weight_history.append(weights)
#
#         if i > 0 and i % resample_every == 0:
#             print("resampling!")
#             theta_particles_norm, weights = resample_particles_simple(theta_particles_norm, weights)
#             model.update(new_weights=torch.from_numpy(weights).double(),
#                          new_particles=torch.unsqueeze(torch.from_numpy(theta_particles_norm), 0).double())
#         else:
#             theta_particles_norm = predict(theta_particles_norm, explore_noise)
#             # weights = update_weights(theta_particles_norm, obj_norm, prior_model,
#             #                          x_eval[i], all_bounds_wt, weights, device)
#             weights = update_weights_ordering(theta_particles_norm,
#                                               y_eval_norm,
#                                               prior_model,
#                                               x_eval,
#                                               device,
#                                               np.array(tuning_config.weights))
#             model.update(new_weights=torch.from_numpy(weights))
#         print(
#             f"particle with highest weight is {denormalize(theta_particles_norm[np.argmax(weights)], all_bounds_wt[num_gains:])}")
#
#         # x = torch.cat([x0_torch, torch.unsqueeze(torch.from_numpy(x_eval[:i + 1, :]), 0).to(device)], dim=1)
#         # y = torch.cat([y, torch.from_numpy(y_eval[i]).view(1, 1, len(robot.perf_metric_names())).to(device)], dim=1)
#         print(
#             f"on iteration {i}, the observed value was {np.dot(obj_norm, tuning_config.weights)}.....best is now {cur_best_obj_norm}")
#         step_log_dict["Evaluation/y_eval"] = np.dot(obj_norm, tuning_config.weights)
#         step_log_dict["Evaluation/best_y_eval"] = cur_best_obj_norm
#         if wandb_run is not None:
#             particle_mean = np.sum(denormalize(theta_particles_norm, all_bounds_wt[num_gains:]) * weights.reshape(-1, 1), axis=0)
#             particle_spread = np.var(denormalize(theta_particles_norm, all_bounds_wt[num_gains:]) * weights.reshape(-1, 1), axis=0)
#             theta_names = robot.get_theta_names()[theta_idxs]
#             for j, name in enumerate(theta_names):
#                 step_log_dict["Sys-Id/particle_mean_"+name] = particle_mean[j]
#                 step_log_dict["Sys-Id/particle_var_"+name] = particle_spread[j]
#             wandb_run.log(step_log_dict)
#     if wandb_run is not None:
#         wandb_run.log({"Evaluated Gains": wandb.Table(columns=robot.get_gain_names()[robot.gains_to_optimize].tolist(),
#                                                       data=denormalize(x_eval, all_bounds_wt[:num_gains]).tolist())})
#
#     particle_results = ParticleResults(np.array(particle_history), np.array(weight_history))
#     return Result(best_gain,
#                   cur_best_obj,
#                   obj_values,
#                   x0, y0, x_eval,
#                   y_eval,
#                   model.state_dict(),
#                   robot,
#                   particle_results,
#                   np.array(test_errors),
#                   np.array(test_variances))


def botorch_optimize_particles_hierarchical(robot: Robot,
                                            tuning_config: ExperimentConfig,
                                            x_test: np.ndarray = None,
                                            y_test: np.ndarray = None,
                                            wandb_run=None):
    # Values for logging.
    obj_values = []
    best_obj_values = []
    acq_values = []
    best_gain = []
    particle_history = []
    weight_history = []
    test_errors = []
    test_variances = []
    cur_best_obj = -np.inf
    cur_best_obj_norm = -100
    cur_best_noisy_obj = -np.inf
    cur_best_noisy_obj_norm = -100

    # Parameters for optimization and sys-id
    opt_params = tuning_config.opt_params
    num_evals = opt_params.num_evals
    num_gains = len(robot.gains_to_optimize)
    theta_idxs = np.array(tuning_config.robot_info.thetas_to_sweep, dtype=int)
    num_priors = opt_params.num_priors_to_use
    # params_to_slice = np.append(robot.gains_to_optimize,
    #                             np.array(theta_idxs, dtype=int) + len(robot.gains_to_optimize))
    resample_every = tuning_config.sys_id.resample_every
    explore_noise = tuning_config.sys_id.explore_noise
    x_eval = np.zeros((num_evals, len(robot.gains_to_optimize)))
    y_eval = np.zeros((num_evals, len(robot.perf_metric_names())))
    y_eval_norm = np.zeros((num_evals, len(robot.perf_metric_names())))

    # Loading priors
    x0 = np.load(os.path.join(robot.prior_dir, tuning_config.prior_x0_file))[:num_priors]
    y0 = np.load(os.path.join(robot.prior_dir, tuning_config.prior_y0_file))[:num_priors]
    with open(os.path.join(robot.prior_dir, tuning_config.scaler_file), 'rb') as f:
        y_scaler = pickle.load(f)
    # prior_model, prior_mll, x0_torch, y0_torch, y_scaler, all_bounds_wt, device = prepare_and_train_prior(x0, y0, robot,
    #                                                                                                       theta_idxs,
    #                                                                                                       y_scaler=y_scaler,
    #                                                                                                       cpu=True,
    #                                                                                                       variational=False)
    network, x0_torch, y0_torch, y_scaler, all_bounds_wt, device = prepare_and_train_prior_nn(x0, y0, robot, theta_idxs,
                                                                                              y_scaler=y_scaler, cpu=False)
    gain_bounds = all_bounds_wt[:num_gains, :]
    x0_torch = build_all_x(x0[:, robot.gains_to_optimize], [], gain_bounds)

    gain_bounds_norm = torch.from_numpy(np.array([[0, 1] for j in range(x0_torch.size(2))], dtype=float).T)

    transform = ScalarizedPosteriorTransform(weights=torch.from_numpy(np.array(tuning_config.weights)))

    # Particle filter stuff
    prior_weights = np.ones(x0.shape[0]) * 1 / x0.shape[0]
    theta_particles_norm = normalize(x0[:, num_gains + theta_idxs], all_bounds_wt[num_gains:])
    theta_particles_norm, prior_weights = resample_particles_simple(theta_particles_norm, prior_weights,
                                                                    num=tuning_config.sys_id.num_particles)
    # add noise to remove initial duplicates
    theta_particles_norm = predict(theta_particles_norm, tuning_config.sys_id.explore_noise)
    theta_particles_torch = torch.unsqueeze(torch.from_numpy(theta_particles_norm), 0).double()
    # sliced_prior_model = WeightedSliceGP(prior_model, weights=torch.from_numpy(prior_weights),
    #                                      theta_particles=theta_particles_torch, x_dim=num_gains, device=device)
    sliced_prior_model = NNSlicePrior(network, y0_torch.size(-1), weights=torch.from_numpy(prior_weights),
                                      theta_particles=theta_particles_torch, device=device)

    true_thetas = robot.get_thetas()[theta_idxs]
    device=torch.device("cpu")  # after training prior, don't really need gpu
    x = torch.zeros(1, 1, num_gains).double().to(device)
    y = torch.zeros(1, 1, len(robot.perf_metric_names())).double().to(device)
    perf_metric_names = robot.perf_metric_names()
    if tuning_config.sys_id.likelihood_method.lower() == "mse":
        MODEL_T = MixedModel
    elif tuning_config.sys_id.likelihood_method.lower() == "ranking":
        MODEL_T = RGPEMixedModel
    else:
        raise NotImplementedError("Sys-ID likelihood method not recognized")
    for i in range(num_evals):
        step_log_dict = {}
        if i == 0:
            model = sliced_prior_model
        else:
            model = MODEL_T(current_model, sliced_prior_model, device)
        if False and x_test is not None and y_test is not None:
            ce_model_mae, model_avg_variance = validate_model(model,
                                                              x_test,
                                                              y_test,
                                                              gain_bounds,
                                                              y_scaler,
                                                              device)
            test_errors.append(ce_model_mae)
            test_variances.append(model_avg_variance)
            for j in range(len(perf_metric_names)):
                step_log_dict["Validation/Test_Err " + perf_metric_names[j]] = ce_model_mae[j]
                step_log_dict["Validation/Test_Var " + perf_metric_names[j]] = model_avg_variance[j]
        # acqf = make_acqf(tuning_config.opt_params, model, cur_best_obj_norm, transform, x)
        sampler = SobolQMCNormalSampler(256)
        qUCB = qUpperConfidenceBound(model, 0.5, sampler, posterior_transform=transform)
        # candidate, acq_value = optimize_acqf(acqf, bounds=gain_bounds_norm, q=1, num_restarts=5, raw_samples=30)
        candidate, acq_value = optimize_acqf(qUCB, q=1, num_restarts=5, bounds=gain_bounds_norm, raw_samples=30)
        # optimize acqf silently moves model to cpu
        # print(f"Acq value was {acq_value}")
        candidate_np = candidate.cpu().detach().numpy()[0]
        x_eval[i] = candidate_np
        obj = robot.evaluate_x(denormalize(candidate_np, gain_bounds), render_override=False)
        if len(tuning_config.noise_std) > 0:
            obj_noise = obj + np.random.randn(*obj.shape) * np.array(tuning_config.noise_std).reshape(obj.shape)
        else:
            obj_noise = obj
        diffs = test_model_fit(model, candidate_np.reshape(1, -1), obj.reshape(1, -1),
                               y_scaler, device, x_already_normalized=True)
        for j, name in enumerate(perf_metric_names):
            step_log_dict["Validation/Model Residual (wrt gt obj) on "+name] = diffs[j]
        y_eval[i] = obj_noise
        obj_noise_norm = y_scaler.transform(obj_noise.reshape(1, -1))
        obj_norm = y_scaler.transform(obj.reshape(1, -1))
        y_eval_norm[i] = obj_noise_norm
        if np.dot(obj_noise_norm, tuning_config.weights) > cur_best_noisy_obj_norm:
            cur_best_obj_norm = np.dot(obj_norm, tuning_config.weights)
            cur_best_noisy_obj_norm = np.dot(obj_noise_norm, tuning_config.weights)
            best_gain = x_eval[i]

        # In practice this is not actually known. Right now just here for logging purposes.
        if np.dot(obj_norm, tuning_config.weights) > cur_best_obj_norm:
            cur_best_obj = np.dot(obj, tuning_config.weights)
            cur_best_obj_norm = np.dot(obj_norm, tuning_config.weights)
        best_obj_values.append(cur_best_obj_norm)
        obj_values.append(obj_noise_norm)  # should maybe save the ground truth as well?
        acq_values.append(acq_value)
        particle_history.append(denormalize(theta_particles_norm, all_bounds_wt[num_gains:]))
        weight_history.append(prior_weights)

        if i > 0 and i % resample_every == 0:
            print("resampling!")
            theta_particles_norm, prior_weights = resample_particles_simple(theta_particles_norm, prior_weights/np.sum(prior_weights))
            sliced_prior_model.update(new_weights=torch.from_numpy(prior_weights).double(),
                                      new_particles=torch.unsqueeze(torch.from_numpy(theta_particles_norm), 0).double())
        elif i >= 2:
            theta_particles_norm = predict(theta_particles_norm, explore_noise)
            model.update_weights(torch.from_numpy(theta_particles_norm),
                                 y_eval_norm[:i+1], x_eval[:i+1], prior_model, device)
        print(
            f"particle with highest weight is {denormalize(theta_particles_norm[np.argmax(prior_weights)], all_bounds_wt[num_gains:])}")

        if i == 0:
            x[0, 0, :] = torch.from_numpy(x_eval[i]).double().to(device)
            y[0, 0, :] = torch.from_numpy(obj_noise_norm).double().to(device)
            current_mll, current_model = initialize_model(x, y)
        else:
            x = torch.cat([x, torch.unsqueeze(torch.from_numpy(x_eval[i:i + 1, :]), 0).double().to(device)], dim=1)
            y = torch.cat([y, torch.from_numpy(obj_noise_norm).view(1, 1, obj_noise_norm.shape[1]).double().to(device)], dim=1)
            current_mll, current_model = initialize_model(x, y, state_dict=current_model.state_dict())
        fit_gpytorch_mll(current_mll)
        print(
            f"on iteration {i}, the observed value was {np.dot(obj_noise_norm, tuning_config.weights)}.....best is now {cur_best_noisy_obj_norm}")
        print(
            f"on iteration {i}, the ground truth value was {np.dot(obj_norm, tuning_config.weights)}.....best gt is now {cur_best_obj_norm}")
        step_log_dict["Evaluation/y_eval_noisy"] = np.dot(obj_noise_norm, tuning_config.weights)
        step_log_dict["Evaluation/y_eval_gt"] = np.dot(obj_norm, tuning_config.weights)
        step_log_dict["Evaluation/best_y_eval_noisy"] = cur_best_noisy_obj_norm
        step_log_dict["Evaluation/best_y_eval_gt"] = cur_best_obj_norm
        if wandb_run is not None:
            particle_weights = prior_weights/np.sum(prior_weights)
            particle_mean = np.sum(denormalize(theta_particles_norm, all_bounds_wt[num_gains:]) * particle_weights.reshape(-1, 1), axis=0)
            particle_spread = np.var(denormalize(theta_particles_norm, all_bounds_wt[num_gains:]) * particle_weights.reshape(-1, 1), axis=0)
            theta_names = robot.get_theta_names()[theta_idxs]
            for j, name in enumerate(theta_names):
                step_log_dict["Sys-Id/particle_mean_"+name] = particle_mean[j]
                step_log_dict["Sys-Id/particle_mean_error_"+name] = true_thetas[j] - particle_mean[j]
                step_log_dict["Sys-Id/particle_var_"+name] = particle_spread[j]
            wandb_run.log(step_log_dict)
    if wandb_run is not None:
        wandb_run.log({"Evaluated Gains": wandb.Table(columns=robot.get_gain_names()[robot.gains_to_optimize].tolist(),
                                                      data=denormalize(x_eval, all_bounds_wt[:num_gains]).tolist())})
    particle_results = ParticleResults(np.array(particle_history), np.array(weight_history))
    return Result(best_gain,
                  cur_best_obj,
                  obj_values,
                  x0, y0, x_eval,
                  y_eval,
                  model.state_dict(),
                  robot,
                  particle_results,
                  np.array(test_errors),
                  np.array(test_variances))


# def botorch_optimize_mle(robot: Robot,
#                          theta_hat: np.ndarray,
#                          tuning_config: ExperimentConfig):
#     obj_values = []
#     best_obj_values = []
#     acq_values = []
#     best_gain = []
#     theta_hats = []
#
#     cur_best_obj = -10
#     opt_params = tuning_config.opt_params
#     num_evals = opt_params.num_evals
#     num_gains = len(robot.gains_to_optimize)
#     theta_idxs = tuning_config.robot_info.thetas_to_sweep
#     theta_idxs_allshift = np.array(theta_idxs, dtype=int) + len(robot.get_gain_names())
#     theta_idxs_shift = np.array(theta_idxs, dtype=int) + len(robot.gains_to_optimize)
#     num_thetas = len(theta_idxs)
#     num_priors = opt_params.num_priors_to_use
#     params_to_slice = np.append(robot.gains_to_optimize,
#                                 np.array(theta_idxs, dtype=int) + len(robot.gains_to_optimize))
#
#     if not opt_params.use_priors:
#         num_points = opt_params.n_initial_points
#         raise NotImplementedError("Can't do MLE estimation on theta without priors. \
#                                   Falling back to normal optimization over prior.")
#     else:
#         x0 = np.load(os.path.join(robot.prior_dir, tuning_config.prior_x0_file))[:num_priors]
#         y0 = np.load(os.path.join(robot.prior_dir, tuning_config.prior_y0_file))[:num_priors]
#
#     prior_model, prior_mll, x0_torch, y0_torch, y_scaler, all_bounds = prepare_and_train_prior(x0, y0, robot,
#                                                                                                theta_idxs)
#     theta_hat_norm = normalize_theta(theta_hat, all_bounds)
#     all_bounds_torch = torch.from_numpy(all_bounds.T)
#     all_bounds_norm = torch.from_numpy(np.array([[0, 1] for j in range(x0_torch.size(2))], dtype=float).T)
#
#     x = x0_torch
#     y = y0_torch
#     model_ucb = prior_model
#     mll_ucb = prior_mll
#     transform = ScalarizedPosteriorTransform(weights=torch.from_numpy(np.array(tuning_config.weights)))
#
#     x_eval = np.zeros((num_evals, len(robot.gains_to_optimize)))
#     y_eval = np.zeros((num_evals, len(robot.perf_metric_names())))
#
#     theta_dists = []
#     true_theta = robot.get_thetas()[theta_idxs]
#     print(f"TRUE THETAS (SECRET) = {true_theta}")
#     for i in range(num_evals):
#         mll_ucb, model_ucb = initialize_model(x, y, state_dict=model_ucb.state_dict())
#         fit_gpytorch_mll(mll_ucb)
#         UCB = UpperConfidenceBound(model_ucb, beta=0.1, posterior_transform=transform)
#         # This also only works for continuous theta idxs...
#         fixed_thetas = {i: theta_hat_norm[i - num_gains] for i in range(num_gains, num_gains + num_thetas)}
#         candidate, acq_value = optimize_acqf(UCB, bounds=all_bounds_norm, q=1, num_restarts=1, raw_samples=20,
#                                              fixed_features=fixed_thetas)
#         candidate_np = candidate.detach().numpy()[0, :num_gains]
#         x_eval[i] = candidate_np
#         obj = robot.evaluate_x(denormalize(candidate_np, all_bounds[:num_gains, :]))
#         y_eval[i] = y_scaler.transform(obj.reshape(1, obj.shape[0]))
#         '''
#         plot_conditional_prior(x0,
#                                y0,
#                                robot,
#                                theta_idxs,
#                                weights=np.array(tuning_config.weights),
#                                # condition_on=denormalize(candidate.detach().numpy()[0], all_bounds),
#                                condition_on=denormalize(np.append(x_eval[i], theta_hat), all_bounds),  # conditional points are with true thetas
#                                annotation=(denormalize(candidate.detach().numpy()[0], all_bounds), y_eval[i]))  # Annotation points are with theta_hat
#         '''
#         # plot_log_probs_wrt_theta(prior_model, x_eval[:i+1,:], y_eval[:i+1, :], theta_hat_norm, 1, robot.get_theta_names(), robot.perf_metric_names())
#         if np.dot(y_eval[i], tuning_config.weights) > cur_best_obj:
#             cur_best_obj = np.dot(y_eval[i], tuning_config.weights)
#             best_gain = x_eval[i]
#         best_obj_values.append(cur_best_obj)
#         obj_values.append(obj)
#         acq_values.append(acq_value)
#         theta_hats.append(theta_hat)
#         # Do update of theta_hat from MLE
#         theta_hat = denormalize_theta(
#             update_theta_mle(prior_model, x_eval[:i + 1, :], y_eval[:i + 1, :], normalize_theta(theta_hat, all_bounds),
#                              step_size=tuning_config.sys_id.mle_learning_rate,
#                              num_steps=tuning_config.sys_id.mle_num_steps),
#             all_bounds)
#         print(f"updated theta_hat = {theta_hat}")
#         theta_dists.append(np.linalg.norm(normalize_theta(theta_hat - true_theta, all_bounds)))
#         # prepare the new dataset with the new theta_hat
#         x = build_all_x(x0[:, params_to_slice], theta_hat, all_bounds, eval_gains=x_eval[:i + 1, :])
#         y = torch.cat([y, torch.from_numpy(y_eval[i]).view(1, 1, len(robot.perf_metric_names()))], dim=1)
#
#         print(
#             f"on iteration {i}, the observed value was {np.dot(y_eval[i], tuning_config.weights)}.....best is now {cur_best_obj}")
#     plt.plot(theta_dists)
#     plt.title("distance of theta_hat from true theta (normalized in each dimension)")
#     plt.show()
#     return Result(best_gain, cur_best_obj, best_obj_values, x0, y0, x_eval, y_eval, model_ucb.state_dict(),
#                   robot=Robot, sys_id_results=MLEResults(np.array(theta_hats)))
