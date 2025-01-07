import os
import torch
from omegaconf import OmegaConf
import numpy as np

from learned_ctrlr_opt.eval.eval_utils import load_kf_and_scalers, load_task_scaler
from learned_ctrlr_opt.meta_learning.lsr_net import LSRBasisNet, LSRBasisNet_encoder
from learned_ctrlr_opt.meta_learning.basis_kf import kalman_step, last_layer_prediction_uncertainty_aware
from learned_ctrlr_opt.utils.dataset_utils import unpp_metrics, pp_metrics
from learned_ctrlr_opt.opt.random_search import random_search


class OCCAMModel(object):
    def __init__(self,
                 experiment_cfg:OmegaConf,
                 path_header: str =None,
                 task_input: bool =False):


        self.experiment_cfg = experiment_cfg
        self.kf_cfg = OmegaConf.load(os.path.join(path_header, experiment_cfg.kf_ckpt_dir, "config.yaml"))
        self.kf_network, self.gain_scaler, self.history_scaler, self.metric_scaler = load_kf_and_scalers(experiment_cfg)
        if task_input:
            # need to change this variable name to something more reasonable-sounding.
            self.ref_track_scaler = load_task_scaler(self.kf_cfg, flatten=self.kf_cfg.ref_track_per_term_scaling)

        # no reason to ever use mps at the moment.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.weights = self.kf_network.last_layer_prior
        self.sigma = torch.mm(self.kf_network.last_layer_prior_cov_sqrt,
                         torch.t(self.kf_network.last_layer_prior_cov_sqrt)).float().to(self.device)
        self.Q = torch.mm(self.kf_network.Q_sqrt, torch.t(self.kf_network.Q_sqrt)).float().to(self.device)
        self.R = torch.mm(self.kf_network.R_sqrt, torch.t(self.kf_network.R_sqrt)).float().to(self.device)

        # keep memory of previous trials.
        self.previous_optima = None
        self.previous_performances = None

        self.previous_weights = None
        self.previous_sigmas = None


    def reset_model(self):
        self.weights = self.kf_network.last_layer_prior
        self.sigma = torch.mm(self.kf_network.last_layer_prior_cov_sqrt,
                              torch.t(self.kf_network.last_layer_prior_cov_sqrt)).float().to(self.device)
        self.Q = torch.mm(self.kf_network.Q_sqrt, torch.t(self.kf_network.Q_sqrt)).float().to(self.device)
        self.R = torch.mm(self.kf_network.R_sqrt, torch.t(self.kf_network.R_sqrt)).float().to(self.device)
        self.previous_optima = None
        self.previous_performances = None
        self.previous_weights = None
        self.previous_sigmas = None
        pass

    def save_occam_state(self, path: str = "temp_occam_state.pt", save_all: bool = False):
        if save_all:
            d = {"weights": self.previous_weights,
                 "sigma": self.previous_sigmas,
                 "Q": self.Q,
                 "R": self.R}
        else:
            d = {"weights": self.previous_weights[-1].unsqueeze(0),
                 "sigma": self.previous_sigmas[-1].unsqueeze(0),
                 "Q": self.Q,
                 "R": self.R}

        torch.save(d, path)

    def load_occam_state(self, path: str = "temp_occam_state.pt"):
        d = torch.load(path)
        self.previous_weights = d["weights"]
        self.sigma = d["sigma"]
        self.Q = d["Q"]
        self.R = d["R"]
        self.weights = self.previous_weights[-1]
        self.sigma = self.previous_sigmas[-1]

        self.previous_optima = None
        self.previous_performances = None

    def adapt_model(self,
                    gains: np.ndarray,
                    observed_perf: np.ndarray,
                    task_input:np.ndarray=None,
                    history: np.ndarray = None):
        inputs_torch = self.preprocess_inputs(gains, task_input, history)
        phi = self.kf_network(inputs_torch.float().to(self.device).unsqueeze(0)).squeeze()
        observation_scaled = self.metric_scaler.transform(pp_metrics(observed_perf, self.kf_cfg).reshape(1, -1))
        target = torch.from_numpy(observation_scaled).float().to(self.device).squeeze()
        with torch.no_grad():
            weights, sigma, K = kalman_step(self.weights, self.sigma, target.float().to(self.device), phi, self.Q, self.R)

        if self.previous_weights is None:
            self.previous_weights = weights
        else:
            self.previous_weights = torch.cat([self.previous_weights, weights.unsqueeze(0).cpu()], dim=0)

        if self.previous_sigmas is None:
            self.previous_sigmas = sigma
        else:
            self.previous_sigmas = torch.cat([self.previous_sigmas, sigma.unsqueeze(0).cpu()], dim=0)

        self.weights = weights
        self.sigma = sigma

    def predict(self, gains, task_input=None, history=None):
        inputs_torch = self.preprocess_inputs(gains, task_input, history)
        best_y_mean, best_y_sigma = last_layer_prediction_uncertainty_aware(inputs_torch,
                                                                            self.kf_network,
                                                                            self.weights,
                                                                            self.sigma)
        best_y_mean = best_y_mean.cpu().detach().numpy()
        best_y_sigma = best_y_sigma.cpu().detach().numpy()

        # also need to un-preprocess metrics, if done during training.
        best_y_unscaled = unpp_metrics(self.metric_scaler.inverse_transform(best_y_mean.reshape(1, -1)))
        return best_y_mean, best_y_sigma, best_y_unscaled

    def optimize_random_search(self, task_input=None, history=None):
        gain_dim = len(self.kf_cfg.gains_to_optimize)
        def eval_fn(x, cost_weights, sigma_weight):
            q = x.size(0)
            cost_weights = torch.from_numpy(cost_weights).float().to(self.device)
            ys, sigmas = last_layer_prediction_uncertainty_aware(x.float().to(self.device),
                                                                 self.kf_network,
                                                                 self.weights,
                                                                 self.sigma)
            cost_weights_batch = cost_weights.repeat(q).reshape((q, cost_weights.shape[-1])).to(self.device)
            mean_losses = torch.sum(ys * cost_weights_batch, dim=-1)
            variances = torch.zeros(x.size(0)).to(self.device)
            for j in range(x.size(0)):
                inter = torch.mm(sigmas[j], cost_weights.unsqueeze(-1))
                variances[j] = torch.mm(torch.t(cost_weights.unsqueeze(-1)), inter)
            losses = mean_losses + sigma_weight * variances
            return losses, ys, variances
        fixed_inputs = self.preprocess_inputs(gains=None, task_input=task_input, history=history)
        if self.previous_optima is not None:
            # where does gain_dim come from?
            perturbed_optima = [self.previous_optima]
            for num_to_add in range(5):
                noisy_task_input_data = torch.clip(self.previous_optima + torch.randn(
                    self.previous_optima.shape) * self.experiment_cfg.sample_noise, 0, 1)
                perturbed_optima.append(noisy_task_input_data)
            exploit_samples = torch.cat(perturbed_optima, dim=0)
        else:
            exploit_samples = None
        results = random_search(eval_fn,
                                self.experiment_cfg.num_search_samples,
                                self.experiment_cfg.cost_weights,
                                gain_dim,
                                self.device,
                                fixed_inputs,
                                self.experiment_cfg.batch_size,
                                self.experiment_cfg.sigma_weight,
                                exploit_samples)
        best_gain_unscaled = self.gain_scaler.inverse_transform(
            results[0].detach().cpu().numpy()
        )
        return best_gain_unscaled, results

    def preprocess_inputs(self, gains=None, task_input=None, history=None):
        inputs = []
        if gains is not None:
            if len(gains.shape) < 2:
                gains_scaled = self.gain_scaler.transform(gains.reshape(1, -1))
            else:
                gains_scaled = self.gain_scaler.transform(gains)
            inputs.append(gains_scaled)
        if history is not None:
            traj_lim = history[-self.kf_cfg.history_length:]
            traj_lim = np.expand_dims(traj_lim, 0)
            traj_rs = traj_lim.reshape(-1, traj_lim.shape[-1])
            traj_scaled = self.history_scaler.transform(traj_rs)
            traj_flat = torch.from_numpy(np.squeeze(traj_scaled.reshape(traj_lim.shape[0], -1)))
            inputs.append(traj_flat)
        if task_input is not None:
            shape = (-1, 1) if self.kf_cfg.ref_track_per_term_scaling else (1, -1)
            ref_scaled = torch.from_numpy(self.ref_track_scaler.transform(task_input.reshape(shape)))
            inputs.append(ref_scaled)
        if len(inputs) == 0:
            raise ValueError("No inputs to preprocess!")
        inputs_torch = torch.cat(inputs, dim=-1)
        return inputs_torch

    def get_cost_from_perf(self, raw_perf):
        observation_scaled = self.metric_scaler.transform(pp_metrics(raw_perf, self.kf_cfg).reshape(1, -1))
        return np.dot(observation_scaled, self.experiment_cfg.cost_weights)
