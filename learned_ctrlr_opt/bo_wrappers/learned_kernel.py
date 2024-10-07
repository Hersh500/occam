import numpy as np
import torch
from learned_ctrlr_opt.utils.dataset_utils import normalize, denormalize
from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import Kernel, ScaleKernel, MaternKernel
from gpytorch.priors import GammaPrior
from botorch.models.utils import validate_input_scaling
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.models.exact_gp import ExactGP

from learned_ctrlr_opt.encoder.traj_encoder import TrajectoryAE



class DeepKernelGP(SingleTaskGP):
    def __init__(self, train_X, train_Y,
                 kernel_transform,
                 kernel_out_size,
                 likelihood=None,
                 covar_module=None,
                 mean_module=None,
                 outcome_transform=None,
                 input_transform=None):

        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            train_Y, _ = outcome_transform(train_Y)
        self._validate_tensor_args(X=transformed_X, Y=train_Y)
        ignore_X_dims = getattr(self, "_ignore_X_dims_scaling_check", None)
        validate_input_scaling(
            train_X=transformed_X, train_Y=train_Y, ignore_X_dims=ignore_X_dims
        )
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        train_X, train_Y, _ = self._transform_tensor_args(X=train_X, Y=train_Y)
        if likelihood is None:
            noise_prior = GammaPrior(1.1, 0.05)
            noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
            likelihood = GaussianLikelihood(
                noise_prior=noise_prior,
                batch_shape=self._aug_batch_shape,
                noise_constraint=GreaterThan(
                    MIN_INFERRED_NOISE_LEVEL,
                    transform=None,
                    initial_value=noise_prior_mode,
                ),
            )
        else:
            self._is_custom_likelihood = True
        ExactGP.__init__(
            self, train_inputs=train_X, train_targets=train_Y, likelihood=likelihood
        )
        if mean_module is None:
            mean_module = ConstantMean(batch_shape=self._aug_batch_shape)
        self.mean_module = mean_module
        if covar_module is None:
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=kernel_out_size,
                    batch_shape=self._aug_batch_shape,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ),
                batch_shape=self._aug_batch_shape,
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
            self._subset_batch_dict = {
                "likelihood.noise_covar.raw_noise": -2,
                "mean_module.raw_constant": -1,
                "covar_module.raw_outputscale": -1,
                "covar_module.base_kernel.raw_lengthscale": -3,
            }
        self.covar_module = covar_module
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self.to(train_X)
        self.kernel_transform = kernel_transform
        self.kernel_transform.eval()

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                x_transformed = self.kernel_transform(x)
        else:
            x_transformed = self.kernel_transform(x)
        return super().forward(x_transformed)

    def train(self, mode=True):
        super().train(mode)
        # Don't want to train the kernel transform jointly
        self.kernel_transform.eval()
        return self


def optimize_single_task(test_agent,
                         num_trials,
                         num_random_trials,
                         seed,
                         metric_scaler,
                         cost_weights,
                         device,
                         gain_scaler,
                         metric_idxs,
                         x_prior=None,
                         y_prior=None):
    np.random.seed(seed)
    num_features = len(test_agent.gains_to_optimize)
    cur_best_obj = -10
    cur_best_gain = []
    best_objs = []
    state_dict = None
    norm_box_bound = torch.from_numpy(np.array([[gain_scaler.feature_range[0], gain_scaler.feature_range[1]] for i in range(num_features)]).T).double()
    best_gains_denorm = []
    if x_prior is None and y_prior is None:
        x_exp = np.zeros((num_trials, num_features))
        y_exp = np.zeros((num_trials, 1))
        data_idx = 0
    elif x_prior is None or y_prior is None or x_prior.shape[0] != y_prior.shape[0]:
        raise ValueError("Mismatch between x_prior and y_prior!")
    else:
        x_exp = np.zeros((num_trials+x_prior.shape[0], num_features))
        y_exp = np.zeros((num_trials+y_prior.shape[0], y_prior.shape[1]))
        x_exp[:x_prior.shape[0],:] = x_prior
        y_exp[:y_prior.shape[0],:] = y_prior
        data_idx = x_prior.shape[0]
        x_exp_torch = torch.from_numpy(x_exp[:data_idx, :]).unsqueeze(0).double()
        y_exp_torch = torch.from_numpy(y_exp[:data_idx, :]).unsqueeze(0).double()
    for step in range(num_trials):
        if step < num_random_trials and data_idx == 0:
            candidate_np = np.random.rand(num_features) * (gain_scaler.feature_range[1] - gain_scaler.feature_range[0]) + gain_scaler.feature_range[0]
        else:
            if step < num_random_trials:
                print(f"Not using random trials since we have prior data")
            new_model = SingleTaskGP(x_exp_torch, y_exp_torch).to(device)
            new_mll = ExactMarginalLogLikelihood(new_model.likelihood, new_model).to(device)
            if state_dict is not None:
                new_model.load_state_dict(state_dict)
            fit_gpytorch_mll(new_mll)
            state_dict = new_model.state_dict()
            acqf = ExpectedImprovement(new_model, best_f=cur_best_obj)
            sampler = SobolQMCNormalSampler(256)
            if step < num_random_trials:
                start_idx = 0
            else:
                start_idx = data_idx
            # acqf = qNoisyExpectedImprovement(new_model.cpu(), X_baseline=x_exp_torch.cpu(), sampler=sampler)
            candidate, acq_value = optimize_acqf(acqf,
                                                 bounds=norm_box_bound,
                                                 q=1,
                                                 num_restarts=5,
                                                 raw_samples=200)
            candidate_np = candidate.detach().cpu().numpy()
        # metrics = test_agent.evaluate_x(denormalize(candidate_np, gain_bounds))
        metrics = test_agent.evaluate_x(gain_scaler.inverse_transform(candidate_np.reshape(1, -1)).flatten())[metric_idxs]
        # print(metrics)
        metrics_norm = metric_scaler.transform(metrics.reshape(1, -1)).flatten()
        # metrics_norm = metrics
        obj = np.dot(cost_weights, metrics_norm)
        if obj > cur_best_obj:
            cur_best_obj = obj
            cur_best_gain = gain_scaler.inverse_transform(candidate_np.reshape(1, -1))
        best_gains_denorm.append(cur_best_gain)
        best_objs.append(cur_best_obj)
        # print(f"Step {step}: tried gains were {denormalize(candidate, all_bounds)}")
        print(f"Step {step}: normalized tried gains were {candidate_np}")
        print(f"Step {step}: obtained obj was {obj}")
        x_exp[step+data_idx, :] = candidate_np
        y_exp[step+data_idx, :] = obj
        x_exp_torch = torch.from_numpy(x_exp[:step + 1 + data_idx, :]).unsqueeze(0).double()
        y_exp_torch = torch.from_numpy(y_exp[:step + 1 + data_idx, :]).unsqueeze(0).double()
    return new_model, new_mll, best_objs, best_gains_denorm, x_exp[data_idx:,:], y_exp[data_idx:,:]


def optimize_learned_kernel_single_task(test_agent,
                                        extrinsic_features,
                                        kernel_network,
                                        metric_scaler,
                                        num_trials,
                                        num_random_trials,
                                        seed,
                                        cost_weights,
                                        device,
                                        gain_scaler,
                                        metric_idxs,
                                        x_prior = None,
                                        y_prior = None,
                                        latent_size = None):
    if extrinsic_features is None:
        extrinsic_features = np.array([])
    if latent_size is None:
        latent_size = len(cost_weights)
    np.random.seed(seed)
    cur_best_obj = -10
    cur_best_gain = []
    best_objs = []
    best_gains_denorm = []
    num_gains = len(test_agent.gains_to_optimize)
    num_features = num_gains + len(extrinsic_features)
    if x_prior is None and y_prior is None:
        x_exp = np.zeros((num_trials, num_features))
        y_exp = np.zeros((num_trials, 1))
        data_idx = 0
    elif x_prior is None or y_prior is None or x_prior.shape[0] != y_prior.shape[0]:
        raise ValueError("Mismatch between x_prior and y_prior!")
    else:
        x_exp = np.zeros((num_trials+x_prior.shape[0], num_features))
        y_exp = np.zeros((num_trials+y_prior.shape[0], y_prior.shape[1]))
        x_exp[:x_prior.shape[0],:] = x_prior
        y_exp[:y_prior.shape[0],:] = y_prior
        data_idx = x_prior.shape[0]
        x_exp_torch = torch.from_numpy(x_exp[:data_idx,:]).unsqueeze(0).double()
        y_exp_torch = torch.from_numpy(y_exp[:data_idx,:]).unsqueeze(0).double()
    state_dict = None
    norm_box_bound = torch.from_numpy(np.array([[gain_scaler.feature_range[0], gain_scaler.feature_range[1]] for i in range(num_features)]).T).double()
    kernel_network.cpu()
    if len(extrinsic_features) != 0:
        fixed_features = {i:extrinsic_features[i-num_gains] for i in range(num_gains, num_features)}
    else:
        fixed_features = None
    for step in range(num_trials):
        if step < num_random_trials and data_idx == 0:
            print(f"Trial {step}: Using random gain")
            candidate_np_gains = np.random.rand(num_gains) * (gain_scaler.feature_range[1] - gain_scaler.feature_range[0]) + gain_scaler.feature_range[0]
            candidate = np.append(candidate_np_gains, extrinsic_features)
            candidate = torch.from_numpy(candidate.reshape(1, -1)).double()
        else:
            if step < num_random_trials:
                print(f"Not using random trials because we have prior data!")
            new_model = DeepKernelGP(x_exp_torch, y_exp_torch, kernel_transform=kernel_network, kernel_out_size=latent_size).to(device)
            new_mll = ExactMarginalLogLikelihood(new_model.likelihood, new_model).to(device)
            if state_dict is not None:
                new_model.load_state_dict(state_dict)
            # for param in new_model.covar_module.parameters():
            #     print(f"param before fit: {param}")
            fit_gpytorch_mll(new_mll)
            # for param in new_model.covar_module.parameters():
            #     print(f"param after fit: {param}")
            state_dict = new_model.state_dict()
            acqf = ExpectedImprovement(new_model, best_f=cur_best_obj)
            if step < num_random_trials:
                start_idx = 0
            else:
                start_idx = data_idx
            # sampler = SobolQMCNormalSampler(256)
            # acqf = qNoisyExpectedImprovement(new_model.cpu(), X_baseline=x_exp_torch[:,start_idx:].cpu(), sampler=sampler)
            candidate, acq_value = optimize_acqf(acqf,
                                                 bounds=norm_box_bound,
                                                 q=1,
                                                 num_restarts=5,
                                                 raw_samples=200,
                                                 fixed_features=fixed_features)
        candidate_np = candidate.detach().cpu().numpy()
        candidate_np_gains = candidate_np[0,:num_gains]
        # metrics = test_agent.evaluate_x(denormalize(candidate_np_gains, gain_bounds))
        metrics = test_agent.evaluate_x(gain_scaler.inverse_transform(candidate_np_gains.reshape(1, -1)).flatten())[metric_idxs]
        # print(f"Raw metrics were {metrics}")
        # print(metrics)
        metrics_norm = metric_scaler.transform(metrics.reshape(1, -1)).flatten()
        # metrics_norm = metrics
        # print(f"normalized metrics: {metrics_norm}")
        # print(f"Candidate: {candidate}")
        # print(f"expected metrics from neural net: {kernel_network(candidate).cpu().detach().numpy()}")
        obj = np.dot(cost_weights, metrics_norm)
        print(f"Trial {step}: obtained obj {obj}")
        if obj > cur_best_obj:
            cur_best_obj = obj
            cur_best_gain = gain_scaler.inverse_transform(candidate_np_gains.reshape(1, -1))
        best_gains_denorm.append(cur_best_gain)
        best_objs.append(cur_best_obj)
        # print(f"Step {step}: tried gains were {denormalize(candidate, all_bounds)}")
        # print(f"Step {step}: normalized tried gains were {candidate_np}")
        # print(f"Step {step}: obtained obj was {obj}")
        x_exp[step+data_idx,:num_gains] = candidate_np_gains
        x_exp[step+data_idx,num_gains:] = extrinsic_features
        y_exp[step+data_idx,:] = obj
        x_exp_torch = torch.from_numpy(x_exp[:step+1+data_idx,:]).unsqueeze(0).double()
        y_exp_torch = torch.from_numpy(y_exp[:step+1+data_idx,:]).unsqueeze(0).double()
    return new_model, new_mll, best_objs, best_gains_denorm, x_exp[data_idx:,:], y_exp[data_idx:,:]

def random_search(test_agent,
                  num_trials,
                  seed,
                  metric_scaler,
                  cost_weights,
                  gain_scaler,
                  metric_idxs):
    np.random.seed(seed)
    num_features = len(test_agent.gains_to_optimize)
    cur_best_obj = -10
    cur_best_gain = []
    best_objs = []
    best_gains_denorm = []
    x_exp = np.zeros((num_trials, num_features))
    y_exp = np.zeros((num_trials, 1))
    data_idx = 0
    for step in range(num_trials):
        candidate_np = np.random.rand(num_features) * (gain_scaler.feature_range[1] - gain_scaler.feature_range[0]) + \
                       gain_scaler.feature_range[0]
        metrics = test_agent.evaluate_x(gain_scaler.inverse_transform(candidate_np.reshape(1, -1)).flatten())[metric_idxs]
        metrics_norm = metric_scaler.transform(metrics.reshape(1, -1)).flatten()
        obj = np.dot(cost_weights, metrics_norm)
        if obj > cur_best_obj:
            cur_best_obj = obj
            cur_best_gain = gain_scaler.inverse_transform(candidate_np.reshape(1, -1))
        best_gains_denorm.append(cur_best_gain)
        best_objs.append(cur_best_obj)
        # print(f"Step {step}: tried gains were {denormalize(candidate, all_bounds)}")
        print(f"Step {step}: normalized tried gains were {candidate_np}")
        print(f"Step {step}: obtained obj was {obj}")
        x_exp[step+data_idx, :] = candidate_np
        y_exp[step+data_idx, :] = obj
    return best_objs, best_gains_denorm, x_exp[data_idx:,:], y_exp[data_idx:,:]