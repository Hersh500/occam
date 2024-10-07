from learned_ctrlr_opt.systems.robots import Robot
from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler
import h5py
from datetime import datetime
import os


@dataclass
class HartmannInputs:
    x1: float = 0.5
    x2: float = 0.5
    x3: float = 0.5
    x4: float = 0.5
    x5: float = 0.5
    x6: float = 0.5

    def get_list(self):
        return np.array([self.x1,
                         self.x2,
                         self.x3,
                         self.x4,
                         self.x5,
                         self.x6])

    def get_names(self):
        return ["x1", "x2", "x3", "x4", "x5", "x6"]

    @staticmethod
    def get_num():
        return 6

    @staticmethod
    def get_bounds():
        return np.array([[0, 1],
                         [0, 1],
                         [0, 1],
                         [0, 1],
                         [0, 1],
                         [0, 1]])

    @staticmethod
    def generate_random(to_randomize=None):
        if to_randomize is None:
            to_randomize = [i for i in range(HartmannInputs.get_num())]
        params = HartmannInputs().get_list()
        bounds = HartmannInputs.get_bounds()
        for idx in to_randomize:
            params[idx] = np.random.uniform(bounds[idx][0], bounds[idx][1])
        return HartmannInputs(*params)


@dataclass
class HartmannFnParams:
    a1: float = 1.0
    a2: float = 1.0
    a3: float = 2.4
    a4: float = 3.0

    def get_list(self):
        return np.array([self.a1, self.a2, self.a3, self.a4])

    @staticmethod
    def get_bounds():
        return np.array([[0.5, 1.5],
                        [0.6, 1.4],
                        [2.0, 3.0],
                        [2.8, 3.6]])

    @staticmethod
    def get_num():
        return 4

    @staticmethod
    def get_names():
        return np.array(["a1", "a2", "a3", "a4"])

    @staticmethod
    def generate_random(to_randomize=None):
        if to_randomize is None:
            to_randomize = [i for i in range(HartmannFnParams.get_num())]
        params = HartmannFnParams().get_list()
        bounds = HartmannFnParams.get_bounds()
        for idx in to_randomize:
            params[idx] = np.random.uniform(bounds[idx][0], bounds[idx][1])
        return HartmannFnParams(*params)


@dataclass
class HartmannFnParamsTest(HartmannFnParams):
    @staticmethod
    def lower_half_bounds():
        return np.array([[0.5, 1.0],
                         [0.6, 1.0],
                         [2.0, 2.4],
                         [2.8, 3.0]])

    @staticmethod
    def upper_half_bounds():
        return np.array([[1.0, 1.5],
                        [1.2, 1.4],
                        [2.4, 3.0],
                        [3.4, 3.6]])
        

    @staticmethod
    def generate_random(to_randomize=None):
        lower = np.random.randint(2)  # pick between 0, 1
        if lower == 0:
            bounds = HartmannFnParamsTest.lower_half_bounds()
        else:
            bounds = HartmannFnParamsTest.upper_half_bounds()
        if to_randomize is None:
            to_randomize = [i for i in range(HartmannFnParamsTest.get_num())]
        params = HartmannFnParamsTest().get_list()
        for idx in to_randomize:
            params[idx] = np.random.uniform(bounds[idx][0], bounds[idx][1])
        return HartmannFnParamsTest(*params)


# Narrower Training Range, to study extrapolation to OOD tasks
@dataclass
class HartmannFnParamsTrain(HartmannFnParams):
    @staticmethod
    def get_bounds():
        return np.array([[1.0, 1.5],
                         [1.0, 1.2],
                         [2.4, 3.0],
                         [3.0, 3.4]])

    @staticmethod
    def generate_random(to_randomize=None):
        if to_randomize is None:
            to_randomize = [i for i in range(HartmannFnParamsTrain.get_num())]
        params = HartmannFnParamsTrain().get_list()
        bounds = HartmannFnParamsTrain.get_bounds()
        for idx in to_randomize:
            params[idx] = np.random.uniform(bounds[idx][0], bounds[idx][1])
        return HartmannFnParamsTrain(*params)


class Hartmann(Robot):
    def __init__(self, hartmann_params: HartmannFnParams, gains_to_optimize=[0, 1, 2, 3, 4, 5]):
        self.params = hartmann_params
        self.alpha = hartmann_params.get_list()
        self.gains_to_optimize = gains_to_optimize
        self.prior_dir = "priors/hartmann"
        self.test_set_dir = "test_sets/hartmann"
        self.ParamsT = HartmannFnParams
        self.ControllerParamsT = HartmannInputs
        self.A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                           [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                           [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                           [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])

        self.P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                                    [2329, 4135, 8307, 3736, 1004, 9991],
                                    [2348, 1451, 3522, 2883, 3047, 6650],
                                    [4047, 8828, 8732, 5743, 1091, 381]])

    def perf_metric_names(self):
        return ["val"]

    # borrowed from f-pacoh codebase
    # https://github.com/jonasrothfuss/f-pacoh-torch/tree/main
    def evaluate_x(self, x):
        inputs = np.array(HartmannInputs().get_list())
        inputs[self.gains_to_optimize] = x
        assert x.ndim <= 2
        external_sum = 0
        for i in range(4):
            internal_sum = 0
            for j in range(6):
                if x.ndim == 1:
                    internal_sum += self.A[i, j] * (x[j] - self.P[i, j]) ** 2
                else:
                    internal_sum += self.A[i, j] * (x[:, j] - self.P[i, j]) ** 2
            external_sum += self.alpha[i] * np.exp(-internal_sum)
        return external_sum / 3.322368011391339

    def get_gain_bounds(self):
        return self.ControllerParamsT.get_bounds()[self.gains_to_optimize]

    def get_theta_bounds(self):
        return self.ParamsT.get_bounds()

    def get_nominal_values(self):
        return np.array(self.ControllerParamsT().get_list() + self.ParamsT.get_list())

    def get_thetas(self):
        return self.params.get_list()

    @staticmethod
    def gather_data(n_points, gains_to_optimize, params_to_sweep):
        X = np.zeros((n_points, HartmannInputs.get_num() + HartmannFnParams.get_num()))
        y = np.zeros((n_points, 1))
        for i in range(n_points):
            thetas = HartmannFnParams.generate_random(params_to_sweep)
            gains = np.array(HartmannInputs.generate_random(gains_to_optimize).get_list())
            val = Hartmann(thetas, gains_to_optimize).evaluate_x(gains[gains_to_optimize])
            X[i,:] = gains
            y[i,:] = val
        scaler = StandardScaler().fit(y)
        return X, y, scaler


def gather_theta_batched_hartmann_data(n_datapoints,
                                       batch_size,
                                       gains_to_randomize,
                                       thetas_to_randomize,
                                       high_level_folder,
                                       params_t=HartmannFnParamsTrain):
    
    assert n_datapoints == int(n_datapoints / batch_size) * batch_size
    
    num_batches = int(n_datapoints / (batch_size))
    intrinsics = np.zeros((num_batches, len(thetas_to_randomize)))
    gains_to_test = np.zeros((num_batches, batch_size, len(gains_to_randomize)))

    subfolder = "hartmann_data_" + datetime.now().strftime("%b_%d_%Y_%H%M") + "/"
    if not os.path.exists(os.path.join(high_level_folder, subfolder)):
        os.makedirs(os.path.join(high_level_folder, subfolder), exist_ok=True)

    with h5py.File(os.path.join(high_level_folder, subfolder, "dataset.hdf5"), 'w') as f:
        f.create_dataset("intrinsics", shape=intrinsics.shape)
        f.create_dataset("gains", shape=gains_to_test.shape)
        f.create_dataset("metrics", shape=(num_batches, batch_size, 1))
        for batch in range(num_batches):
            batch_intrinsic_obj = params_t.generate_random(thetas_to_randomize)
            f["intrinsics"][batch,...] = batch_intrinsic_obj.get_list()
            robot = Hartmann(batch_intrinsic_obj, gains_to_randomize)
            for point in range(batch_size):
                gain = HartmannInputs.generate_random(gains_to_randomize).get_list()[gains_to_randomize]
                metric = robot.evaluate_x(gain)
                f["metrics"][batch,point] = metric
                f["gains"][batch,point] = gain
            print(f"finished batch {batch}")
