from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import gym
import numpy as np
import multiprocessing as mp
import warnings
from sklearn.preprocessing import StandardScaler

# Top down car racing imports
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from learned_ctrlr_opt.systems.car_controller import CarController, CarControllerParams
from learned_ctrlr_opt.systems.car_dynamics import CarParams
from learned_ctrlr_opt.systems.car_racing import CarRacing

from learned_ctrlr_opt.systems.quadrotor2d import Quadrotor2DParams, Quadrotor2DControllerParams
# Quadrotor Imports
try:
    from systems.quadrotor2d import Quadrotor2DSim
    from pydrake.all import (AddMultibodyPlantSceneGraph,
                             DiagramBuilder, Linearize, LinearQuadraticRegulator,
                             Simulator)
except ImportError:
    print("Cannot use pydrake on this system. Quadrotor2d will not work")
    pass


### System Wrapper Classes ###
class Robot(ABC):
    gains_to_optimize = []
    ParamsT: Any
    ControllerParamsT: Any
    prior_dir: str
    test_set_dir: str

    @abstractmethod
    def __init__(self):
        # Intialize any persistent simulations, etc. in here.
        return

    @abstractmethod
    def evaluate_x(self, x):
        # Run evaluation
        # Resets should be handled internally within this method
        return [0]

    def get_gain_names(self):
        return np.array(self.ControllerParamsT.get_names())

    def get_theta_names(self):
        return np.array(self.ParamsT.get_names())

    def get_gain_dim(self):
        return len(self.gains_to_optimize)

    @abstractmethod
    def perf_metric_names(self):
        return []

    @abstractmethod
    def get_gain_bounds(self):
        return []

    @abstractmethod
    def get_theta_bounds(self):
        return []

    @abstractmethod
    def get_nominal_values(self):
        return []


class TopDownCar(Robot):
    def __init__(self, seed, car_params=CarParams(), gains_to_optimize=[], render=False, length=None, max_time=3000):
        if len(gains_to_optimize) == 0:
            self.gains_to_optimize = [i for i in range(0, CarControllerParams.get_num())]
        else:
            self.gains_to_optimize = gains_to_optimize
        self.params = car_params
        self.seed = seed
        self.prior_dir = "priors/topdowncar"
        self.test_set_dir = "test_sets/topdowncar"
        self.ParamsT = CarParams
        self.ControllerParamsT = CarControllerParams
        self.render = render
        self.length = length
        self.max_time = max_time

    def evaluate_x(self, x, render_override=False):
        if isinstance(x, CarControllerParams):
            controller = CarController(x)
        else:
            defaults = np.array(CarControllerParams().get_list())
            defaults[self.gains_to_optimize] = x
            controller = CarController(CarControllerParams(*list(defaults)))
        if isinstance(self.seed, list):
            seed = self.seed[0]
        else:
            seed = self.seed
        if self.render or render_override:
            env = CarRacing(params=self.params, render_mode="human")
        else:
            env = CarRacing(params=self.params)
        perf_metrics, laptime = controller.test_on_track(env, seed=seed, num_steps=self.max_time, length=self.length)
        avgs = np.mean(np.array(perf_metrics), axis=0)
        return np.append(avgs, [laptime])

    def evaluate_x_return_traj(self, x, render_override=False):
        if isinstance(x, CarControllerParams):
            controller = CarController(x)
        else:
            defaults = CarControllerParams().get_list()
            defaults[self.gains_to_optimize] = x
            controller = CarController(CarControllerParams(*list(defaults)))
        if isinstance(self.seed, list):
            seed = self.seed[0]
        else:
            seed = self.seed
        if self.render or render_override:
            env = CarRacing(params=self.params, render_mode="human")
        else:
            env = CarRacing(params=self.params)
        perf_metrics, laptime, traj = controller.test_on_track_return_traj(env, seed=seed, num_steps=self.max_time, length=self.length)
        avgs = np.mean(np.array(perf_metrics), axis=0)
        return np.append(avgs, [laptime]), traj

    @staticmethod
    def perf_metric_names():
        return CarController.metric_names

    def get_gain_bounds(self):
        bounds = np.array(CarControllerParams.get_bounds())[self.gains_to_optimize]
        skopt_bounds = []
        # for bound in bounds:
        # skopt_bounds.append(space.space.Real(bound[0], bound[1]))
        # return skopt_bounds
        return bounds

    def get_gain_dim(self):
        return len(self.gains_to_optimize)

    def get_gain_names(self):
        return np.array(CarControllerParams.get_names())

    def get_theta_bounds(self):
        return np.array(CarParams.get_bounds())

    def get_theta_names(self):
        return np.array(CarParams.get_names())

    def get_nominal_values(self):
        return np.array(CarControllerParams().get_list()+CarParams().get_list())

    def get_track(self):
        env = CarRacing(self.params)
        env.reset(seed=self.seed)
        return [[env.track[i][2], env.track[i][3]] for i in range(self.length)]

    def get_thetas(self):
        if isinstance(self.seed, list):
            warnings.warn("Using the first seed only to compute the curvature, length of track")
            seed = self.seed[0]
        else:
            seed = self.seed
        env = CarRacing(self.params)
        env.reset(seed=seed)
        # self.env.reset(seed=seed)
        if self.length is not None:
            track_lim = self.length
        else:
            track_lim = len(env.track)
        curve = TopDownCar.compute_curvature(np.array([[point[2], point[3]] for point in env.track[:track_lim]]))
        # print(f"curvature = {curve}")
        length = track_lim
        # print(f"length = {length}")
        # since curve, length are only determined at runtime, have to get rid of the placeholders and add the real ones
        return np.array(self.params.get_list()[:-2] + [curve, length])

    @staticmethod
    def compute_curvature(points):
        angs = []
        for i in range(1, points.shape[0]-1):
            ang1 = np.arcsin((points[i][0] - points[i-1][0])/np.linalg.norm(points[i] - points[i-1]))
            ang2 = np.arcsin((points[i+1][0] - points[i][0])/np.linalg.norm(points[i+1] - points[i]))
            angs.append(np.abs(ang1 - ang2))
        return np.mean(angs)

    @staticmethod
    def worker(gains_to_optimize, params_to_sweep, track_seeds, seed, length=None):
        np.random.seed(seed)
        if len(track_seeds) == 0:
            seed = int(np.random.randint(0, 2000))
        else:
            seed = int(np.random.choice(track_seeds))
        thetas = CarParams.generate_random(params_to_sweep)
        gains = CarControllerParams.generate_random(gains_to_optimize)
        env = CarRacing(params=thetas)
        metrics, laptime = CarController(gains=gains).test_on_track(env, num_steps=3000, seed=seed, length=length)
        if length is None:
            track_lim = len(env.track)
        else:
            track_lim = length
        thetas = thetas.get_list()
        thetas[-2] = TopDownCar.compute_curvature(np.array([[point[2], point[3]] for point in env.track[:track_lim]]))
        thetas[-1] = track_lim
        x = np.append(np.array(gains.get_list()),np.array(thetas))
        y = np.append(np.mean(np.array(metrics), axis=0), [laptime])
        return x, y

    @staticmethod
    def gather_data(n_points, gains_to_optimize, params_to_sweep, seeds, length=None):
        X = np.zeros((n_points, CarControllerParams.get_num() + CarParams.get_num()))
        y = np.zeros((n_points, 4))
        if not isinstance(seeds, Iterable):
            seeds = [seeds]
        num_done = 0
        while num_done < n_points:
            num_proc = max(1, mp.cpu_count() - 4)
            p = mp.Pool(processes=num_proc)
            return_val_array = p.starmap(TopDownCar.worker,
                                         [(gains_to_optimize, params_to_sweep, seeds, np.random.randint(0,10000), length) for j in range(num_proc)])
            for i in range(num_proc):
                X[num_done+i,:] = return_val_array[i][0]
                y[num_done+i, :] = return_val_array[i][1]
            num_done += num_proc
        scaler = StandardScaler().fit(y)
        return X, y, scaler

        # for i in range(n_points):
        #     thetas = CarParams.generate_random(params_to_sweep)
        #     gains = CarControllerParams.generate_random(gains_to_optimize)
        #     car_env = CarRacing(params=thetas)
        #     ctrlr = CarController(gains=gains)
        #     if len(seeds) == 0:
        #         seed = int(np.random.randint(0, 2000))
        #     else:
        #         seed = int(np.random.choice(seeds))
        #     try:
        #         metrics, laptime = ctrlr.test_on_track(car_env, num_steps=3000, seed=seed)
        #         avgs = np.mean(np.array(metrics), axis=0)
        #     except KeyboardInterrupt:
        #         break
        #     y[i][0:3] = avgs
        #     y[i][3] = laptime
        #     X[i][0:CarControllerParams.get_num()] = np.array(gains.get_list())
        #     X[i][CarControllerParams.get_num():] = np.array(thetas.get_list())
        #     Last two thetas are determined from track generation.
            # X[i][-2] = TopDownCar.compute_curvature(np.array([[point[2], point[3]] for point in car_env.track]))
            # X[i][-1] = len(car_env.track)
        # return X[:i,:], y[:i,:]


class TopDownCarRandomStartingState(Robot):
    def __init__(self,
                 seed,
                 initial_gain,
                 car_params=CarParams(),
                 gains_to_optimize=[],
                 length=None,
                 initial_length=100,
                 initial_sensor_traj_length=30,
                 max_time=3000):
        if len(gains_to_optimize) == 0:
            self.gains_to_optimize = [i for i in range(0, CarControllerParams.get_num())]
        else:
            self.gains_to_optimize = gains_to_optimize
        self.params = car_params
        self.seed = seed
        self.prior_dir = "priors/topdowncar"
        self.test_set_dir = "test_sets/topdowncar"
        self.ParamsT = CarParams
        self.ControllerParamsT = CarControllerParams
        self.length = length
        self.max_time = max_time

        self.initial_gain = initial_gain
        self.initial_length = initial_length
        self.initial_sensor_traj_length = initial_sensor_traj_length

    def evaluate_x(self, x, render=False):
        if isinstance(x, CarControllerParams):
            controller = CarController(x)
        else:
            defaults = CarControllerParams().get_list()
            defaults[self.gains_to_optimize] = x
            controller = CarController(CarControllerParams(*list(defaults)))

        if isinstance(self.initial_gain, CarControllerParams):
            initial_controller = CarController(self.initial_gain)
        else:
            defaults = CarControllerParams().get_list()
            defaults[self.gains_to_optimize] = self.initial_gain
            initial_controller = CarController(CarControllerParams(*list(defaults)))

        if isinstance(self.seed, list):
            seed = self.seed[0]
        else:
            seed = self.seed
        if render:
            env = CarRacing(params=self.params, render_mode="human")
        else:
            env = CarRacing(params=self.params)

        env.reset(seed=seed)
        # get all the initial stuff
        # should I have a different initial length?
        initial_metrics, initial_laptime, initial_traj = initial_controller.test_on_track_reset_free(env, num_steps=self.max_time, length=self.initial_length)
        initial_traj = initial_traj[:initial_laptime]
        if render:
            print("done with initial part!")
            print(f"initial state is {initial_traj[-1]}")

        # do a check to see if it's fucked...
        if np.abs(initial_traj[-1][2]) < 15:
            perf_metrics, laptime, sensor_traj = controller.test_on_track_reset_free(env, num_steps=self.max_time, length=self.length)
            avgs = np.mean(np.array(perf_metrics), axis=0)
            return True, np.append(avgs, [laptime]), sensor_traj, initial_traj[-self.initial_sensor_traj_length:]
        else:
            return False, None, None, None

    def get_track(self):
        env = CarRacing(self.params)
        env.reset(seed=self.seed)
        track_length = len(env.track)
        return [[env.track[i%track_length][2], env.track[i%track_length][3]] for i in range(self.initial_length, self.initial_length+self.length)]

    def get_gain_bounds(self):
        return CarControllerParams.get_bounds()[self.gains_to_optimize]

    def get_nominal_values(self):
        return []

    def get_theta_bounds(self):
        return CarParams.get_bounds()

    def perf_metric_names(self):
        return CarController.metric_names

    def evaluate_x_initial(self, render=False):
        if isinstance(self.initial_gain, CarControllerParams):
            initial_controller = CarController(self.initial_gain)
        else:
            defaults = CarControllerParams().get_list()
            defaults[self.gains_to_optimize] = self.initial_gain
            initial_controller = CarController(CarControllerParams(*list(defaults)))

        if isinstance(self.seed, list):
            seed = self.seed[0]
        else:
            seed = self.seed
        if render:
            env = CarRacing(params=self.params, render_mode="human")
        else:
            env = CarRacing(params=self.params)

        env.reset(seed=seed)
        # get all the initial stuff
        # should I have a different initial length?
        initial_metrics, initial_laptime, initial_traj = initial_controller.test_on_track_reset_free(env, num_steps=self.max_time, length=self.initial_length)
        initial_traj = initial_traj[:initial_laptime]
        if np.abs(initial_traj[-1][2]) < 15:
            return True, initial_traj[-self.initial_sensor_traj_length:], env
        else:
            return False, initial_traj[-self.initial_sensor_traj_length:], env

    def evaluate_x_post(self, x, env):
        if isinstance(x, CarControllerParams):
            controller = CarController(x)
        else:
            defaults = CarControllerParams().get_list()
            defaults[self.gains_to_optimize] = x
            controller = CarController(CarControllerParams(*list(defaults)))

        perf_metrics, laptime, sensor_traj = controller.test_on_track_reset_free(env, num_steps=self.max_time, length=self.length)
        avgs = np.mean(np.array(perf_metrics), axis=0)
        return np.append(avgs, [laptime]), sensor_traj


# Allows running multiple experiments sequentially
class TopDownCarResetFree(Robot):
    def __init__(self,
                 seed,
                 car_params=CarParams(),
                 gains_to_optimize=[],
                 length=None,
                 max_time=3000,
                 record=False):
        if len(gains_to_optimize) == 0:
            self.gains_to_optimize = [i for i in range(0, CarControllerParams.get_num())]
        else:
            self.gains_to_optimize = gains_to_optimize
        self.params = car_params
        self.seed = seed
        self.prior_dir = "priors/topdowncar_reset_free"
        self.test_set_dir = "test_sets/topdowncar_reset_free"
        self.ParamsT = CarParams
        self.ControllerParamsT = CarControllerParams
        self.length = length
        self.max_time = max_time
        self.record = record

    def evaluate_x(self, x, env=None, render=False, vid_folder=None, vid_prefix=None):
        if env is None:
            if isinstance(self.seed, list):
                seed = self.seed[0]
            else:
                seed = self.seed
            if render and not self.record:
                env = CarRacing(params=self.params, render_mode="human")
            elif render and self.record:
                step_trigger = lambda x: x == 0
                env = gym.wrappers.RecordVideo(CarRacing(params=self.params, render_mode="rgb_array"),
                                               video_folder=vid_folder,
                                               name_prefix=vid_prefix,
                                               step_trigger=step_trigger)
            else:
                env = CarRacing(params=self.params)
            env.reset(seed=seed)

        if isinstance(x, CarControllerParams):
            controller = CarController(x)
        else:
            defaults = CarControllerParams().get_list()
            defaults[self.gains_to_optimize] = x
            controller = CarController(CarControllerParams(*list(defaults)))

        perf_metrics, laptime, sensor_traj = controller.test_on_track_reset_free(env, num_steps=self.max_time, length=self.length)
        avgs = np.mean(np.array(perf_metrics), axis=0)
        return np.append(avgs, [laptime]), sensor_traj, env

    def get_track_from(self, env):
        p = np.array([env.car.hull.position[0], env.car.hull.position[1]])
        track = np.array([[env.track[i][2], env.track[i][3]] for i in range(len(env.track))])
        dists = np.linalg.norm(track-p, axis=1)
        closest_idx = np.argmin(dists)
        # print(f"closest idx is {closest_idx}")
        return np.array([[env.track[i%len(track)][2], env.track[i%len(track)][3]] for i in range(closest_idx, closest_idx+self.length)])

    def get_gain_bounds(self):
        return CarControllerParams.get_bounds()[self.gains_to_optimize]

    def get_nominal_values(self):
        return []

    def get_theta_bounds(self):
        return CarParams.get_bounds()

    def perf_metric_names(self):
        return CarController.metric_names


class Quadrotor2DLQR(Robot):
    def __init__(self, params=Quadrotor2DParams(), gains_to_optimize=[],
                 duration=10.0, init_state=(1, 1)):
        self.params = params
        self.duration = duration
        self.init_state = init_state
        if len(gains_to_optimize) == 0:
            self.gains_to_optimize = [i for i in range(Quadrotor2DControllerParams.get_num())]
        else:
            self.gains_to_optimize = gains_to_optimize
        self.prior_dir = "priors/quadrotor2d"
        self.ParamsT = Quadrotor2DParams
        self.ControllerParamsT = Quadrotor2DControllerParams
        self.test_set_dir = "test_sets/quadrotor2d"

    # discrepancy between types: here, "x" is all the gains, whereas for topdowncar, "x" is only the gains being swept.
    def evaluate_x(self, x, render_override=False):
        if isinstance(x, Quadrotor2DControllerParams):
            lst = x.get_list()
        else:
            lst = np.array(Quadrotor2DControllerParams().get_list())
            lst[self.gains_to_optimize] = x
        def QuadrotorLQR(plant):
            context = plant.CreateDefaultContext()
            context.SetContinuousState(np.zeros([6, 1]))
            plant.get_input_port(0).FixValue(context, plant.mass * plant.gravity / 2. * np.array([1, 1]))

            Q = np.diag([lst[0], lst[1], lst[2], lst[3], lst[4], lst[5]])
            R = np.array([[lst[6], 0], [0, lst[7]]])
            return LinearQuadraticRegulator(plant, context, Q, R)

        builder = DiagramBuilder()
        plant = builder.AddSystem(Quadrotor2DSim(quad_params=Quadrotor2DParams(*self.params.get_list())))
        controller = builder.AddSystem(QuadrotorLQR(plant))
        builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
        builder.Connect(plant.get_output_port(0), controller.get_input_port(0))

        diagram = builder.Build()

        # Set up a simulator to run this diagram
        simulator = Simulator(diagram)
        context = simulator.get_mutable_context()
        context.SetTime(0.)
        context.SetContinuousState(np.array([self.init_state[0], self.init_state[1], 0, 0, 0, 0]))
        simulator.Initialize()
        t = 0.0
        tstep = 0.01
        x = []
        u = []
        while t < self.duration:
            state = context.get_continuous_state_vector().CopyToVector()
            inp = plant.EvalVectorInput(plant.GetMyContextFromRoot(context), 0).CopyToVector()
            u.append(inp)
            x.append(state)
            t += tstep
            simulator.AdvanceTo(t)

        x = np.array(x)
        u = np.array(u)
        # once we have states and inputs available, create metrics on those
        # print(np.nonzero(x[:,0] < 0.1)[0])
        # print(np.nonzero(x[:,0] < 0.9)[0])
        if np.min(x[:,0]) < 0.1:
            rise_time_x = tstep * (np.nonzero(x[:,0] < 0.1)[0][0] - np.nonzero(x[:,0] < 0.9)[0][0])
        else:
            rise_time_x = self.duration
        if np.min(x[:,1]) < 0.1:
            rise_time_y = tstep * (np.nonzero(x[:,1] < 0.1)[0][0] - np.nonzero(x[:,1] < 0.9)[0][0])
        else:
            rise_time_y = self.duration
        overshoot_x = np.clip(np.min(x[:,0]), -20, 0)  # min since we're going from 1 -> 0, overshoot would be negative
        overshoot_y = np.clip(np.min(x[:,1]), -20, 0)  # min since we're going from 1 -> 0, overshoot would be negative
        total_forces = np.linalg.norm(u, axis=0)
        total_velocities = np.linalg.norm(x[:,3:5], axis=0)
        energy = np.sum(total_forces * total_velocities) * tstep
        return np.array([rise_time_x, rise_time_y, overshoot_x, overshoot_y, energy])

    def get_gain_dim(self):
        return len(self.gains_to_optimize)

    def get_gain_names(self):
        return np.array(Quadrotor2DControllerParams.get_names())

    def get_theta_names(self):
        return np.array(Quadrotor2DParams.get_names())

    # not implemented yet
    def get_gain_bounds(self):
        return np.array(Quadrotor2DControllerParams.get_bounds())[self.gains_to_optimize, :]

    def get_nominal_values(self):
        return np.array(Quadrotor2DControllerParams().get_list()+Quadrotor2DParams().get_list())

    def perf_metric_names(self):
        return ["Rise Time X", "Rise Time Y", "Overshoot X", "Overshoot Y", "Energy"]

    @staticmethod
    def get_num_metrics():
        return 5

    def get_thetas(self):
        return np.array(self.params.get_list())

    @staticmethod
    def get_theta_bounds():
        return np.array(Quadrotor2DParams.get_bounds())

    @staticmethod
    def gather_data(n_points, gains_to_optimize, params_to_sweep):
        X = np.zeros((n_points, Quadrotor2DParams.get_num() + Quadrotor2DControllerParams.get_num()))
        y = np.zeros((n_points, len(Quadrotor2DLQR().perf_metric_names())))
        for i in range(n_points):
            thetas = Quadrotor2DParams.generate_random(params_to_sweep)
            gains = Quadrotor2DControllerParams.generate_random(gains_to_optimize)
            metrics = Quadrotor2DLQR(params=thetas, gains_to_optimize=gains_to_optimize).evaluate_x(gains)
            X[i,:] = np.hstack([gains.get_list(), thetas.get_list()])
            y[i] = metrics
            print(f"--------------- i = {i} ---------------")
            print(f"thetas = {thetas}")
            print(f"gains = {gains}")
            print(f"metrics = {metrics}")
        scaler = StandardScaler().fit(y)
        return X, y, scaler
