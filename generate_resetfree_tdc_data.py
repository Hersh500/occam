import numpy as np

from learned_ctrlr_opt.systems.robots import TopDownCarRandomStartingState
from learned_ctrlr_opt.systems.car_controller import CarController, CarControllerParams
from learned_ctrlr_opt.systems.car_dynamics import CarParams, CarParamsTrain, CarParamsTest
from learned_ctrlr_opt.meta_learning.gather_sysid_data import gather_batched_tdc_rss_sysid_data

batch_size = 64
num_batches_per_intrinsic = 1
# n_datapoints = batch_size * 1500
n_datapoints = batch_size * 100
gains_to_randomize = [0, 1, 2, 3, 4, 5]
thetas_to_randomize = [0, 1, 2]
high_level_folder = "priors/tdc_resetfree_sysid_data/"

gather_batched_tdc_rss_sysid_data(n_datapoints,
                                  batch_size,
                                  num_batches_per_intrinsic,
                                  gains_to_randomize,
                                  thetas_to_randomize,
                                  high_level_folder,
                                  initial_length=75,
                                  initial_sensor_traj_length=50,
                                  randomize_tracks=True,
                                  same_track_batches=False,
                                  track_seed=30,
                                  max_time=1500,
                                  params_t=CarParamsTest)
'''

gains_to_randomize = [0, 1, 2, 3, 4, 5]
initial_gain = CarControllerParams()
initial_params = CarParams()
track_seed = 30
initial_length = 75
initial_sensor_traj_length = 25
gain = CarControllerParams().get_list()[gains_to_randomize]

robot = TopDownCarRandomStartingState(seed=track_seed,
                                      initial_gain=initial_gain,
                                      car_params=initial_params,
                                      gains_to_optimize=gains_to_randomize,
                                      length=150,
                                      initial_length=initial_length,
                                      initial_sensor_traj_length=initial_sensor_traj_length,
                                      max_time=1000)

robot.evaluate_x(gain, render=True)
'''
