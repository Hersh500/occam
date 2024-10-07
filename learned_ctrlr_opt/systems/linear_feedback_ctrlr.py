import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Union

from learned_ctrlr_opt.utils.dataset_utils import denormalize
from learned_ctrlr_opt.systems.robots import Robot


# system is default stable but can have nonnegative eigenvalues
@dataclass
class PIDSecondOrderParams:
    a11: float = -1.0
    a12: float = 0.0
    a21: float = 0.0
    a22: float = -1.0
    b11: float = 0.0
    b21: float = 1.0

    def get_list(self):
        return np.array([self.a11, self.a12, self.a21, self.a22, self.b11, self.b21])

    @staticmethod
    def get_bounds():
        return np.array([[-5.0, 5.0],
                         [-1.0, 1.0],
                         [-1.0, 1.0],
                         [-5.0, 5.0],
                         [-2.0, 2.0],
                         [-2.0, 2.0]])

    @staticmethod
    def generate_random(gains_to_randomize):
        random_params = denormalize(np.random.rand(6), PIDSecondOrderParams.get_bounds())
        params = PIDSecondOrderParams().get_list()
        params[gains_to_randomize] = random_params[gains_to_randomize]
        return PIDSecondOrderParams(*params)


@dataclass
class PIDControlParams:
    kp: float = 1.0
    kd: float = 0.5
    ki: float = 0.5

    def get_list(self):
        return np.array([self.kp, self.kd, self.ki])

    @staticmethod
    def get_bounds():
        return np.array([[-5.0, 5.0],
                         [-1.0, 1.0],
                         [-1.0, 1.0],
                         [-5.0, 5.0]])

    @staticmethod
    def generate_random(gains_to_randomize):
        random_params = denormalize(np.random.rand(3), PIDControlParams.get_bounds())
        params = PIDControlParams().get_list()
        params[gains_to_randomize] = random_params[gains_to_randomize]
        return PIDControlParams(*params)


# Implements a second order SISO system w/ PID controller
class PIDDiscreteSecondOrderSys(Robot):
    def __init__(self, params: Union[PIDSecondOrderParams, np.ndarray],
                 sim_dt=0.01,
                 state_noise_std=0.02):
        if isinstance(params, np.ndarray):
            assert len(params) == 6
            self.params = PIDSecondOrderParams(*params)
        else:
            self.params = params
        self.sim_dt = sim_dt
        # should use this noisy value in control.
        self.state_noise_std = state_noise_std

    def evaluate_x(self, g, render=False):
        t_f = 400
        y = np.zeros(t_f)
        y_noisy = np.zeros_like(y)
        x = np.array([0.0, 0.0])
        x_int = 0
        x_d = 0
        t = 0
        r = 1.0
        while t < t_f/self.sim_dt:
            u = g[0] * (r - x[0]) + g[1] * x_d + g[2] * x_int


    def evaluate_x_return_traj(self, x, render=False):
        t_f = 400
        y = np.zeros(t_f)
        y_noisy = np.zeros_like(y)
        x0 = np.array([0.0, 0.0])
        t = 0
        while t < t_f/self.sim_dt:

    def perf_metric_names(self):
        pass

    def get_gain_bounds(self):
        pass

    def get_theta_bounds(self):
        pass

    def get_nominal_values(self):
        pass
