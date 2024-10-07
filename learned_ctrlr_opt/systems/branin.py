from learned_ctrlr_opt.systems.robots import Robot
from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler
import h5py
from datetime import datetime
import os


@dataclass
class BraninInputs:
    x1: float = 0.0
    x2: float = 0.0

    def get_list(self):
        return np.array([self.x1, self.x2])

    def get_names(self):
        return ["x1", "x2"]

    @staticmethod
    def get_num():
        return 2

    @staticmethod
    def get_bounds():
        return np.array([[-5, 10],
                         [0, 15]])

    @staticmethod
    def generate_random(to_randomize=None):
        if to_randomize is None:
            to_randomize = [i for i in range(BraninInputs.get_num())]
        params = BraninInputs().get_list()
        bounds = BraninInputs.get_bounds()
        for idx in to_randomize:
            params[idx] = np.random.uniform(bounds[idx][0], bounds[idx][1])
        return BraninInputs(*params)


A_NOM = 1
B_NOM = 5.1 / (4 * np.pi ** 2)
C_NOM = 5 / np.pi
R_NOM = 6
S_NOM = 10
T_NOM = 1 / (8 * np.pi)


    
@dataclass
class BraninFnParams:
    a: float = A_NOM
    b: float = B_NOM
    c: float = C_NOM
    r: float = R_NOM
    s: float = S_NOM
    t: float = T_NOM

    def get_list(self):
        return np.array([self.a, self.b, self.c, self.r, self.s, self.t])

    @staticmethod
    def get_bounds():
        return np.array([[0.5, 1.5],
                         [0.1, 0.15],
                         [1, 2],
                         [5, 7],
                         [8, 12],
                         [0.03, 0.05]])

    @staticmethod
    def get_num():
        return 6

    @staticmethod
    def get_names():
        return np.array(["a", "b", "c", "r", "s", "t"])

    @staticmethod
    def generate_random(to_randomize=None):
        if to_randomize is None:
            to_randomize = [i for i in range(BraninFnParams.get_num())]
        params = BraninFnParams().get_list()
        bounds = BraninFnParams.get_bounds()
        for idx in to_randomize:
            params[idx] = np.random.uniform(bounds[idx][0], bounds[idx][1])
        return BraninFnParams(*params)


@dataclass
class BraninFnParamsTest(BraninFnParams):
    @staticmethod
    def lower_half_bounds():
        return np.array([[0.5, 0.8],
                         [0.1, 0.11],
                         [1, 1.2],
                         [5, 5.5],
                         [8, 9],
                         [0.03, 0.035]])

    @staticmethod
    def upper_half_bounds():
        return np.array([[1.2, 1.5],
                         [0.13, 0.15],
                         [1.8, 2],
                         [6.5, 7],
                         [11, 12],
                         [0.045, 0.05]])
        

    @staticmethod
    def generate_random(to_randomize=None):
        lower = np.random.randint(2)  # pick between 0, 1
        if lower == 0:
            bounds = BraninFnParamsTest.lower_half_bounds() 
        else:
            bounds = BraninFnParamsTest.upper_half_bounds()
        if to_randomize is None:
            to_randomize = [i for i in range(BraninFnParamsTest.get_num())]
        params = BraninFnParamsTest().get_list()
        for idx in to_randomize:
            params[idx] = np.random.uniform(bounds[idx][0], bounds[idx][1])
        return BraninFnParamsTest(*params)


# Narrower Training Range, to study extrapolation to OOD tasks
@dataclass
class BraninFnParamsTrain(BraninFnParams):
    @staticmethod
    def get_bounds():
        return np.array([[0.8, 1.2],
                         [0.11, 0.13],
                         [1.2, 1.8],
                         [5.5, 6.5],
                         [9, 11],
                         [0.035, 0.045]])

    @staticmethod
    def generate_random(to_randomize=None):
        if to_randomize is None:
            to_randomize = [i for i in range(BraninFnParamsTrain.get_num())]
        params = BraninFnParamsTrain().get_list()
        bounds = BraninFnParamsTrain.get_bounds()
        for idx in to_randomize:
            params[idx] = np.random.uniform(bounds[idx][0], bounds[idx][1])
        return BraninFnParamsTrain(*params)


class Branin(Robot):
    def __init__(self, branin_params: BraninFnParams, gains_to_optimize=[0, 1]):
        self.params = branin_params
        self.gains_to_optimize = gains_to_optimize
        self.prior_dir = "priors/branin"
        self.test_set_dir = "test_sets/branin"
        self.ParamsT = BraninFnParams
        self.ControllerParamsT = BraninInputs

    def perf_metric_names(self):
        return ["val"]

    def evaluate_x(self, x):
        inputs = np.array(BraninInputs().get_list())
        inputs[self.gains_to_optimize] = x
        inter1 = self.params.a * (inputs[1] -
                                  self.params.b * inputs[0]**2 +
                                  self.params.c * inputs[0] - self.params.r)**2
        inter2 = self.params.s * (1 - self.params.t) * np.cos(inputs[0])
        return np.array([inter1 + inter2 + self.params.s])

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
        X = np.zeros((n_points, BraninInputs.get_num() + BraninFnParams.get_num()))
        y = np.zeros((n_points, 1))
        for i in range(n_points):
            thetas = BraninFnParams.generate_random(params_to_sweep)
            gains = np.array(BraninInputs.generate_random(gains_to_optimize).get_list())
            val = Branin(thetas, gains_to_optimize).evaluate_x(gains[gains_to_optimize])
            X[i,:] = gains
            y[i,:] = val
        scaler = StandardScaler().fit(y)
        return X, y, scaler


def gather_theta_batched_branin_data(n_datapoints,
                                     batch_size,
                                     gains_to_randomize,
                                     thetas_to_randomize,
                                     high_level_folder,
                                     params_t=BraninFnParamsTrain):
    
    assert n_datapoints == int(n_datapoints / batch_size) * batch_size
    
    num_batches = int(n_datapoints / (batch_size))
    intrinsics = np.zeros((num_batches, len(thetas_to_randomize)))
    gains_to_test = np.zeros((num_batches, batch_size, len(gains_to_randomize)))

    subfolder = "branin_data_" + datetime.now().strftime("%b_%d_%Y_%H%M") + "/"
    if not os.path.exists(os.path.join(high_level_folder, subfolder)):
        os.makedirs(os.path.join(high_level_folder, subfolder), exist_ok=True)

    with h5py.File(os.path.join(high_level_folder, subfolder, "dataset.hdf5"), 'w') as f:
        f.create_dataset("intrinsics", shape=intrinsics.shape)
        f.create_dataset("gains", shape=gains_to_test.shape)
        f.create_dataset("metrics", shape=(num_batches, batch_size, 1))
        for batch in range(num_batches):
            batch_intrinsic_obj = params_t.generate_random(thetas_to_randomize)
            f["intrinsics"][batch,...] = batch_intrinsic_obj.get_list()
            robot = Branin(batch_intrinsic_obj, gains_to_randomize)
            for point in range(batch_size):
                gain = BraninInputs.generate_random(gains_to_randomize).get_list()[gains_to_randomize]
                metric = robot.evaluate_x(gain)
                f["metrics"][batch,point] = metric
                f["gains"][batch,point] = gain
            print(f"finished batch {batch}")
