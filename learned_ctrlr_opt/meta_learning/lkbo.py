import numpy as np
import torch
import gpytorch
from gpytorch.kernels import Kernel, ScaleKernel, MaternKernel
from gpytorch.priors.torch_priors import GammaPrior

# Maybe the move is just to extract features from the network, and train the network normally.
class DeepMeanModuleNoGrad(torch.nn.Module):
    def __init__(self, nn_module, mean_module, cost_weights):
        super().__init__()
        self.nn_module = nn_module
        self.mean_module = mean_module
        self.cost_weights = cost_weights

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                x_transformed = self.nn_module(x)


class DeepKernelNoGrad(torch.nn.Module):
    def __init__(self, nn_module, kernel_module):
        super().__init__()
        self.nn_module = nn_module
        self.kernel_module = kernel_module

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                x_transformed = self.nn_module(x)
        else:   # some of the botorch optimizers need gradients from the network
            x_transformed = self.nn_module(x)
        return self.kernel_module(x_transformed)

    def train(self, mode=True):
        super().train(mode)
        # Don't want to train the kernel transform jointly
        self.kernel_transform.eval()
        return self


# copied from https://docs.gpytorch.ai/en/stable/examples/06_PyTorch_NN_Integration_DKL/index.html
class DeepGPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_network, latent_dim):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = get_covar_module(latent_dim)
        self.feature_extractor = feature_network

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        with torch.no_grad():
            projected_x = self.feature_extractor(x)
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)  # needs to be projected dim
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_covar_module(num_dims):
    module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=num_dims,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
    return module


def train_dk_gp(train_x: object, train_y: object, network: object, n_iters: object, latent_dim: object, device: object) -> object:
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = DeepGPRegressionModel(train_x, train_y, likelihood, network, latent_dim).to(device)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model).to(device)

    optimizer = torch.optim.Adam([
        {'params': gp_model.covar_module.parameters()},
        {'params': gp_model.mean_module.parameters()},
        {'params': gp_model.likelihood.parameters()},
    ], lr=0.001)

    for i in range(n_iters):
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = gp_model(train_x.to(device))
        # Calc loss and backprop derivatives
        loss = -mll(output, train_y.to(device))
        loss.backward()
        optimizer.step()
    return gp_model, mll
