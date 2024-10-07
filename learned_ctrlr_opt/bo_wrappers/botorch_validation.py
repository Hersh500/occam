from typing import Union, Type

import numpy as np
import torch
from botorch.models import SingleTaskGP
from sklearn.preprocessing import StandardScaler

from learned_ctrlr_opt.bo_wrappers.botorch_models import WeightedSliceGP, MixedModel
from learned_ctrlr_opt.utils.dataset_utils import normalize


def test_model_fit(model: Union[SingleTaskGP, WeightedSliceGP],
                   X_test_np: np.ndarray,
                   y_test_np: np.ndarray,
                   y_scaler: Type[StandardScaler],
                   device: torch.device,
                   bounds: np.ndarray = None,
                   x_already_normalized: bool = False):
    if not x_already_normalized:
        X_test_norm = normalize(X_test_np, bounds)
    else:
        X_test_norm = X_test_np
    X_test_torch = torch.unsqueeze(torch.from_numpy(X_test_norm), 0).to(device)
    out_distr = model(X_test_torch, debug=False)
    out_var = out_distr.variance.squeeze(0).cpu().detach().numpy().reshape(1, -1)
    out_var = y_scaler.inverse_transform(out_var)
    y_out = y_scaler.inverse_transform(out_distr.loc.squeeze(0).cpu().detach().numpy().T)
    # print(f"mean absolute error as a percentage of each metric is {np.mean(100*np.abs(y_out - y_test_np)/(y_test_np+1e-4), axis=0)}")
    diff = y_out - y_test_np  # Don't get absolute error, so we can see if it's underpredicting or overpredicting.
    # print("---- Model evaluation ----")
    # print(f"Absolute error on each test point, in each metric, was {diff}")
    # print(f"GP's confidence in this test point was {out_var}")
    # print(f"The differences as a percentage of the values was {100 * diff/y_test_np}")
    # print("--------------------------")
    return diff.flatten()


def validate_model(model: Union[SingleTaskGP, WeightedSliceGP, MixedModel],
                   x_test: np.ndarray,
                   y_test: np.ndarray,
                   gain_bounds: np.ndarray,
                   y_scaler: StandardScaler,
                   device: torch.device):
    x_test_norm = normalize(x_test, gain_bounds)
    # Do b-batches instead of q-batches
    # x_test_torch = torch.unsqueeze(torch.from_numpy(x_test_norm), 1).double().to(device)
    # x_test_torch = x_test_torch.view(x_test_norm.shape[0], 1, 1, x_test_norm.shape[1])
    y_test_norm = y_scaler.transform(y_test)

    # Can't do whole test set at once due to GPU mem issues. Instead do a few at a time
    model_mean = np.zeros(y_test_norm.shape)
    model_variance = np.zeros(y_test_norm.shape)
    feature_dim = x_test_norm.shape[1]
    for i in range(0, x_test_norm.shape[0]):
        model_distr = model(torch.from_numpy(x_test_norm[i,:]).view(1, 1, feature_dim).double().to(device))
        model_mean[i] = model_distr.mean.cpu().detach().numpy()[0,:,0]
        if i < 0:
            print(f"model_mean = {model_mean[i]}")
            print(f"test_pt = {y_test_norm[i]}")
        model_variance[i] = model_distr.variance.cpu().detach().numpy()[0,:,0]

    ce_model_mae = np.mean(np.abs(model_mean - y_test_norm), axis=0)
    model_avg_variance = np.mean(model_variance, axis=0)
    return ce_model_mae, model_avg_variance
