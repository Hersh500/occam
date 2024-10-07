import torch
import numpy as np
from typing import Union, Optional
from dataclasses import dataclass

# from botorch.models import SingleTaskGP, HeteroskedasticSingleTaskGP, SingleTaskVariationalGP
# from botorch.fit import fit_gpytorch_mll
# from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from learned_ctrlr_opt.systems.robots import Robot
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp

from learned_ctrlr_opt.utils.dataset_utils import normalize, denormalize


# Utility functions


# Assumes theta is stacked AFTER the x in all_bounds
def normalize_theta(thetas, all_bounds):
    try:
        num_thetas = thetas.shape[1]
    except IndexError:
        num_thetas = len(thetas)
    return normalize(thetas, all_bounds[-num_thetas:, :])

# Assumes theta is stacked AFTER the x in all_bounds
# thetas: np.array: (num_points, num_thetas)
def denormalize_theta(thetas, all_bounds):
    try:
        num_thetas = thetas.shape[1]
    except IndexError:
        num_thetas = len(thetas)
    return denormalize(thetas, all_bounds[-num_thetas:, :])


def initialize_model(train_x, train_obj, state_dict=None, obj_noise=None, variational=False):
    if obj_noise is None:
        # print(f"Creating new model. Train_x.shape is {train_x.size()}, train obj size is {train_obj.size()}")
        if not variational:
            new_model = SingleTaskGP(train_x, train_obj).to(train_x.device)
            new_mll = ExactMarginalLogLikelihood(new_model.likelihood, new_model).to(train_x.device)
        else:
            new_model = SingleTaskVariationalGP(train_x[0,:,:], train_Y=train_obj[0,:,:],
                                                inducing_points=min(train_x.size(-2), 200)).to(train_x.device)
            new_mll = VariationalELBO(new_model.likelihood, new_model.model, num_data=train_x.shape[1])
    else:
        new_model = HeteroskedasticSingleTaskGP(train_x, train_obj, train_Yvar=obj_noise)
        new_mll = ExactMarginalLogLikelihood(new_model.likelihood, new_model)
    # load state dict if it is passed
    if state_dict is not None:
        new_model.load_state_dict(state_dict)
    return new_mll, new_model


# takes a np array of gains and a single estimate of theta. normalizes all of them
# and converts into a torch tensor appropriate for botorch: (1, num_points, gains+thetas)
# theta_hat: np.array(num_thetas)
# gains: np.array(num_points, num_gains)
def build_all_x(orig_datapoints, theta_hat, all_bounds, eval_gains=None):
    if eval_gains is not None:
        num_thetas = theta_hat.shape[0]
        num_gains = eval_gains.shape[1]
        all_x = np.zeros((orig_datapoints.shape[0] + eval_gains.shape[0], num_gains+num_thetas))
        all_x[:orig_datapoints.shape[0],:] = orig_datapoints
        all_x[orig_datapoints.shape[0]:,:num_gains] = eval_gains
        all_x[orig_datapoints.shape[0]:,num_gains:] = theta_hat
    else:
        all_x = orig_datapoints
    # unsqueeze is to add a batch dimension for botorch
    return torch.unsqueeze(torch.from_numpy(normalize(all_x, all_bounds)), 0)


def prepare_and_train_prior(x0, y0, robot, params_to_sweep,
                            y0_noise=None, y_scaler=None, cpu=False,
                            variational=False, state_dict=None):
    params_to_slice = np.append(robot.gains_to_optimize,
                                np.array(params_to_sweep, dtype=int) + len(robot.get_gain_names()))
    all_bounds = np.vstack((robot.get_gain_bounds(),
                            robot.get_theta_bounds()[params_to_sweep]))

    if cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    x0_torch = build_all_x(x0[:, params_to_slice], [], all_bounds).to(device)
    if y_scaler is None:
        y_scaler = StandardScaler().fit(y0)
    y0 = y_scaler.transform(y0)
    y0_torch = torch.unsqueeze(torch.from_numpy(y0), 0).to(device)
    mll, model = initialize_model(x0_torch, y0_torch, obj_noise=y0_noise,
                                  variational=variational, state_dict=state_dict)
    print("Beginning prior training...")
    if variational:
        fit_gpytorch_mll_torch(mll)  # just always use this? Is the performance that much worse than the clearinghouse?
    else:
        fit_gpytorch_mll(mll)
    print("Done with prior training.")
    return model, mll, x0_torch, y0_torch, y_scaler, all_bounds, device



# Seed is used to generate ctrl parameters, NOT the track info
def generate_data(n_initial_points,
                  robot,
                  seed=None):
    X = np.zeros((n_initial_points, len(robot.gains_to_optimize)))
    y = np.zeros((n_initial_points, len(robot.perf_metric_names())))
    if seed is not None:
        np.random.seed(seed)
    for i in range(n_initial_points):
        random_gains = robot.ControllerParamsT.generate_random(robot.gains_to_optimize)
        # I wonder how this works when robot.evaluate_x sets the seed...
        metrics = robot.evaluate_x(np.array(random_gains.get_list())[robot.gains_to_optimize])
        X[i] = np.array(random_gains.get_list())[robot.gains_to_optimize]
        y[i] = metrics
    return X, y

# Generate data from a single robot in a parallel way
def generate_data_parallel(n_points, robot):
    X = np.zeros((n_points, len(robot.gains_to_optimize)))
    y = np.zeros((n_points, len(robot.perf_metric_names())))
    num_done = 0
    num_proc = max(1, mp.cpu_count() - 4)
    p = mp.Pool(processes=num_proc)
    num_per = int(n_points/num_proc)
    return_val_array = p.starmap(generate_data,
                                 [(num_per, robot, np.random.randint(0, 10000)) for _ in range(num_proc)])
    for i in range(num_proc):
        X[i*num_per:(i+1)*num_per,:] = return_val_array[i][0]
        y[i*num_per:(i+1)*num_per,:] = return_val_array[i][1]
    scaler = StandardScaler().fit(y)
    return X, y, scaler


def generate_data_sobol(n_points, robot, length=None):
    from scipy.stats import qmc
    closest_power = np.ceil(np.log2(n_points))
    sampler = qmc.Sobol(d=len(robot.gains_to_optimize))
    samples = sampler.random_base2(m=closest_power)


# Theta particles and weights over time
@dataclass
class ParticleResults:
    theta_particles: np.ndarray
    weights: np.ndarray


@dataclass
class MLEResults:
    theta_hats: np.ndarray


@dataclass
class Result:
    best_gain: list
    best_obj: float
    obj_values: list
    x0: Optional[np.ndarray]
    y0: Optional[np.ndarray]
    x_eval: np.ndarray
    y_eval: np.ndarray
    model_dict: Optional[dict]
    robot: Robot
    sys_id_results: Optional[Union[ParticleResults, MLEResults]]
    test_set_mae: Optional[np.ndarray]
    test_set_pred_variance: Optional[np.ndarray]

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
