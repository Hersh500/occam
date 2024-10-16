import numpy as np
from learned_ctrlr_opt.systems.mob_locomotion import gather_batched_mob_loco_sysid_data, Go1HiddenParamsTrain, Go1BehaviorParams, Go1HiddenParamsTest

batch_size = 64
num_batches_per_intrinsic = 1
n_datapoints = batch_size * 100
# n_datapoints = batch_size * 10
gains_to_randomize = [0, 1, 2, 3, 4, 6, 9]  # removed pitch
thetas_to_randomize = [0, 1, 2, 3]
t_f = 4
initial_sensor_traj_length_tsteps = 25

high_level_folder = "priors/mob_locomotion_sysid_data/"

gather_batched_mob_loco_sysid_data(n_datapoints,
                                   batch_size,
                                   num_batches_per_intrinsic,
                                   gains_to_randomize,
                                   thetas_to_randomize,
                                   high_level_folder,
                                   initial_sensor_traj_length_tsteps,
                                   randomize_cmds=True,
                                   t_f=t_f,
                                   params_t=Go1HiddenParamsTest)
