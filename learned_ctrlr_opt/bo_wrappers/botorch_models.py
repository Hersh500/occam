import torch
from torch.nn import Sequential, Linear, ReLU, Sigmoid
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from botorch.models import SingleTaskGP, SingleTaskVariationalGP
from botorch.models.model import Model
from botorch.posteriors.deterministic import DeterministicPosterior
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from gpytorch.models import GP
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from linear_operator.operators import DenseLinearOperator
from typing import Union, Any, Optional, List
from learned_ctrlr_opt.bo_wrappers.sys_id import update_weights_ordering, update_weights
from learned_ctrlr_opt.bo_wrappers.botorch_utils import build_all_x
from sklearn.preprocessing import StandardScaler


class WeightedSliceGP(GP, BatchedMultiOutputGPyTorchModel):
    def __init__(self, full_prior: Union[SingleTaskGP, SingleTaskVariationalGP],
                 weights: torch.Tensor,
                 theta_particles: torch.Tensor,
                 x_dim: int,
                 device: torch.device):

        # slice out the thetas from the training data? So the shapes and everything are correct.
        super(WeightedSliceGP, self).__init__()
        self.device = device
        self.weights = weights.to(device)
        self.theta_particles = theta_particles.to(device)
        self.full_prior = full_prior.eval().to(device)
        self.x_dim = x_dim
        self._num_outputs = full_prior.num_outputs
        self._input_batch_shape = full_prior._input_batch_shape
        self.to(device)


    def update(self, new_weights=None, new_particles=None):
        if new_particles is not None:
            self.theta_particles = new_particles.to(self.device)
            if new_weights is None:
                raise TypeError("If updating particles, new_weights cannot be None.")
        if new_weights is not None:
            self.weights = new_weights.to(self.device)

    def num_outputs(self) -> int:
        return self.full_prior.num_outputs


    def forward(self, X: torch.Tensor, debug=False) -> torch.distributions.MultivariateNormal:
        # print(f"X is of size, {X.size()}")
        num_thetas = self.theta_particles.size(-2)
        theta_features = self.theta_particles.size(-1)
        # then the output at EACH datapoint X should be the sum of the conditional priors.
        # support arbitrary number of batch dims...
        # full_tensor = torch.zeros((X.size(0), X.size(1) * num_thetas, X.size(2) + theta_features)).double()
        shape = list(X.size())
        n_dims = len(shape)
        shape[-1] += theta_features
        shape[-2] *= num_thetas
        full_tensor = torch.zeros(shape)
        # expand n, d dims to match desired shape
        # bigg_tensor[:,:,:self.x_dim] = torch.tile(X, (1, 1, self.theta_particles.size(1), 1))
        full_tensor[...,:self.x_dim] = torch.repeat_interleave(X, num_thetas, dim=n_dims-2)
        full_tensor[...,self.x_dim:] = torch.tile(self.theta_particles, (1, X.size(n_dims-2), 1))
        # each num_datapoints elements of bigg_tensor are now the X's for the same
        # print(f"bigg tensor shape is {full_tensor.size()}")
        # full_tensor = torch.unsqueeze(full_tensor, 1)
        # output = self.full_prior.posterior(full_tensor.double())
        output = self.full_prior(full_tensor.double())
        # print(f"output mean shape:{output.mean.shape}")
        # print(f"output covariance shape{output.covariance_matrix.shape}")
        # TODO(hersh500): Need to match batch-shape here as well for arbitrary # of batches.
        # coincidentally though, I think it works out in the UCB case here since q = n. Should keep in mind for future.
        # print(f"output cov shape is {output.variance.shape}")
        # print(f"output mean shape is {output.mean.shape}")
        output_mean = torch.zeros(X.size(0), self.num_outputs(), X.size(-2)).double()
        weights_repeat = torch.tile(self.weights, (X.size(-2),))
        weights_matrix = torch.outer(weights_repeat, weights_repeat)
        output_cov = torch.zeros(X.size(0), self.num_outputs(), X.size(-2), X.size(-2)).double()
        weighted_covs = output.covariance_matrix * weights_matrix.to(output.mean.device)
        # print(f"prior covariance matrix shape = {output.covariance_matrix.shape}")
        # print(f"weighted_covs = {weighted_covs}")
        for i in range(X.size(n_dims-2)):
            loc = output.loc[:,:,i*num_thetas:(i+1)*num_thetas]
            weighted_sum = torch.sum(loc * self.weights.to(loc.device), dim=2)
            output_mean[:,:,i] = weighted_sum
            for j in range(X.size(n_dims-2)):
                output_cov[:,:,i,j] = torch.sum(weighted_covs[...,:,i*num_thetas:(i+1)*num_thetas,j*num_thetas:(j+1)*num_thetas], dim=(-1, -2))
        # print(f"covariance matrix = {output_cov}")
        # print(f"variances: {np.diagonal(output_cov.detach().cpu().numpy())}")
        return MultivariateNormal(mean=output_mean,
                                  covariance_matrix=DenseLinearOperator(output_cov))

# Wrapper class for mixing old and new data.
class MixedModel(GP, BatchedMultiOutputGPyTorchModel):
    def __init__(self,
                 new_data_model: SingleTaskGP = None,
                 prior_slice_model: WeightedSliceGP = None,
                 device: torch.device = torch.device("cpu")):
        super(MixedModel, self).__init__()
        self.new_data_model = new_data_model
        self.prior_model = prior_slice_model
        if prior_slice_model is not None:
            self._num_outputs = prior_slice_model.num_outputs()
            self._input_batch_shape = prior_slice_model._input_batch_shape
            self.prior_model.to(device)
        if new_data_model is not None:
            self.new_data_model.to(device)
        self.device = device

    # x has shape (b1 x ... x bm, q, d)
    def forward(self, x: torch.Tensor, debug=False):
        # print(f"within mixedModel, x.device is {x.device}")
        new_model_out = self.new_data_model(x.to(self.device))
        prior_model_out = self.prior_model(x.to(self.device))

        # Take a weighted version of the mean:
        new_var = new_model_out.variance.to(self.device)
        prior_var = prior_model_out.variance.to(self.device)
        # weight both based on variance of the other...?
        mean_out = (prior_var * new_model_out.mean.to(self.device) + new_var * prior_model_out.mean.to(self.device))/(prior_var + new_var)
        weights = np.array([-0.5, -0.1, -0.3])
        if debug:
            print(f"New Variance is {new_var.cpu().detach().numpy()}")
            print(f"Prior Variance is {prior_var.cpu().detach().numpy()}")
            print(f"Output mean is {mean_out.cpu().detach().numpy()}")
            print(f"Output covariance is {new_model_out.covariance_matrix}")
            # print(f"UCB for this would be is {np.dot(mean_out.cpu().detach().numpy().flatten(), weights) + np.expand_dims(weights, -2) @ new_model_out.covariance_matrix.cpu().detach().numpy().flatten() @ np.expand_dims(weights, -1)}")
        # print(mean_out.shape)
        return MultivariateNormal(mean_out.cpu(), new_model_out.lazy_covariance_matrix.cpu())

    # def update_weights(self, theta_particles_norm, prior_weights):
    #     self.prior_model.update(prior_weights, theta_particles_norm)

    def update_weights(self, theta_particles_norm,
                       y_eval_norm,
                       x_eval,
                       full_prior_model,
                       device):
        prior_weights = update_weights(theta_particles_norm, y_eval_norm[-1],
                                       full_prior_model, x_eval[-1],
                                       self.prior_model.weights.cpu().detach().numpy(),
                                       device)
        self.prior_model.update(torch.from_numpy(prior_weights), theta_particles_norm)


class RGPEMixedModel(GP, BatchedMultiOutputGPyTorchModel):
    def __init__(self,
                 new_data_model: SingleTaskGP = None,
                 prior_slice_model: WeightedSliceGP = None,
                 device: torch.device = torch.device("cpu"),
                 target_weight: float = 0.0):
        super(RGPEMixedModel, self).__init__()
        self.new_data_model = new_data_model
        self.prior_model = prior_slice_model
        if prior_slice_model is not None:
            self._num_outputs = prior_slice_model.num_outputs()
            self._input_batch_shape = prior_slice_model._input_batch_shape
            self.prior_model.to(device)
        if new_data_model is not None:
            self.new_data_model.to(device)
        self.device = device
        self.target_weight = target_weight
        self.cached_preds = None

    # x has shape (b1 x ... x bm, q, d)
    def forward(self, x: torch.Tensor, debug=False):
        new_model_out = self.new_data_model.to(x)(x)
        prior_model_out = self.prior_model.to(x)(x)

        # Take a weighted version of the mean:
        new_var = new_model_out.variance.to(self.device)
        prior_var = prior_model_out.variance.to(self.device)
        # weight both based on variance of the other...?
        # by construction, the target weight and prior weights should sum to 1
        mean_out = self.target_weight * new_model_out.mean.to(self.device) + prior_model_out.mean.to(self.device)
        if debug:
            print(f"New Variance is {new_var.cpu().detach().numpy()}")
            print(f"Prior Variance is {prior_var.cpu().detach().numpy()}")
            print(f"Output mean is {mean_out.cpu().detach().numpy()}")
            print(f"Output covariance is {new_model_out.covariance_matrix}")
            # print(f"UCB for this would be is {np.dot(mean_out.cpu().detach().numpy().flatten(), weights) + np.expand_dims(weights, -2) @ new_model_out.covariance_matrix.cpu().detach().numpy().flatten() @ np.expand_dims(weights, -1)}")
        # print(mean_out.shape)
        return MultivariateNormal(mean_out.cpu(), new_model_out.lazy_covariance_matrix.cpu())

    def update_weights(self, theta_particles_norm,
                       y_eval_norm,
                       x_eval,
                       full_prior_model,
                       device):
        prior_weights, target_weight, preds = update_weights_ordering(theta_particles_norm, y_eval_norm,
                                                                      full_prior_model, x_eval,
                                                                      device, self.cached_preds)
        self.cached_preds = preds
        self.target_weight = target_weight
        self.prior_model.update(torch.from_numpy(prior_weights), theta_particles_norm)


# How much preprocessing shuold be done in __init__ vs before __init__ vs. after __init__?

def create_network(in_size, out_size):
    return Sequential(
        Linear(in_size, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 64),
        Sigmoid(),
        Linear(64, 16),
        Sigmoid(),
        Linear(16, out_size)
    )


class NNSlicePrior(Model):
    def __init__(self, network, out_size, weights, theta_particles, device):
        super(NNSlicePrior, self).__init__()
        self.out_size = out_size
        self.network = network
        self.weights = weights
        self.theta_particles = theta_particles
        self.device = device
        self.network.to(device)

    def num_outputs(self) -> int:
        return self.out_size

    def batch_shape(self) -> torch.Size:
        return torch.Size([1])

    def update(self, new_weights=None, new_particles=None):
        if new_particles is not None:
            self.theta_particles = new_particles.to(self.device)
            if new_weights is None:
                raise TypeError("If updating particles, new_weights cannot be None.")
        if new_weights is not None:
            self.weights = new_weights.to(self.device)

    def forward(self, x):
        num_thetas = self.theta_particles.size(-2)
        theta_features = self.theta_particles.size(-1)
        x_dim = x.size(-1)
        shape = list(x.size())
        n_dims = len(shape)
        shape[-1] += theta_features
        shape[-2] *= num_thetas
        full_tensor = torch.zeros(shape)
        full_tensor[..., :x_dim] = torch.repeat_interleave(x, num_thetas, dim=n_dims-2)
        full_tensor[..., x_dim:] = torch.tile(self.theta_particles, (1, x.size(n_dims-2), 1))
        output = self.network(full_tensor.double().to(self.device))
        output_mean = torch.zeros(x.size(0), self.num_outputs(), x.size(-2)).double()
        for i in range(x.size(n_dims-2)):
            loc = output[...,:,i*num_thetas:(i+1)*num_thetas]
            weighted_sum = torch.sum(loc * self.weights.unsqueeze(-1).to(output.device), dim=1)
            output_mean[:,:,i] = weighted_sum
        return output_mean

    def posterior(self,
                  X: torch.Tensor,
                  output_indices: Optional[List[int]] = None,
                  observation_noise: bool = False,
                  posterior_transform = None,
                  **kwargs: Any):
        return DeterministicPosterior(self(X))


def prepare_and_train_prior_nn(x0, y0, robot, params_to_sweep, y_scaler=None, cpu=False, ne=300):
    params_to_slice = np.append(robot.gains_to_optimize,
                                np.array(params_to_sweep, dtype=int) + len(robot.get_gain_names()))
    all_bounds = np.vstack((robot.get_gain_bounds(),
                            robot.get_theta_bounds()[params_to_sweep]))

    if cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    x0_torch = build_all_x(x0[:, params_to_slice], [], all_bounds).double().to(device)
    if y_scaler is None:
        y_scaler = StandardScaler().fit(y0)
    y0 = y_scaler.transform(y0)
    y0_torch = torch.unsqueeze(torch.from_numpy(y0), 0).double().to(device)
    dataset = TensorDataset(x0_torch, y0_torch)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=32)
    network = create_network(x0_torch.size(-1), y0_torch.size(-1)).to(device)
    network.double().train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    print("Beginning prior training...")
    for i in range(ne):
        avg_loss = 0
        for j, data in enumerate(dataloader):
            x_batch, y_batch = data
            optimizer.zero_grad()
            predictions = network(x_batch)
            loss = criterion(predictions, y_batch)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"In epoch {i}, avg loss per batch was {avg_loss/len(dataloader)}")
    print("Done with prior training.")
    return network, x0_torch, y0_torch, y_scaler, all_bounds, device
