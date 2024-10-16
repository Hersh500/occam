import numpy as np
from learned_ctrlr_opt.meta_learning.gather_sysid_data import gather_batched_tdc_sysid_data
from learned_ctrlr_opt.systems.car_dynamics import CarParams, CarParamsTrain, CarParamsTest

batch_size = 64
num_batches_per_intrinsic = 1
# n_datapoints = batch_size * 1500
n_datapoints = batch_size * 100
gains_to_randomize = [0, 1, 2, 3, 4, 5]
thetas_to_randomize = [0, 1, 2]
high_level_folder = "priors/tdc_sysid_data/"

gather_batched_tdc_sysid_data(n_datapoints,
                              batch_size,
                              num_batches_per_intrinsic,
                              gains_to_randomize,
                              thetas_to_randomize,
                              high_level_folder,
                              randomize_tracks=True,
                              same_track_batches=True,
                              track_seed=42,
                              params_t=CarParamsTest)
