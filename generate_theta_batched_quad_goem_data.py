import numpy as np
from learned_ctrlr_opt.sysid.gather_sysid_data import gather_batched_quadrotor_sysid_data
from learned_ctrlr_opt.systems.quadrotor_geom import ThreeDCircularTraj_fixed

batch_size = 64
num_batches_per_intrinsic = 1
n_datapoints = batch_size * 1000
# n_datapoints = batch_size * 5
gains_to_randomize = [0, 1, 2, 3, 4, 5, 6, 7]
thetas_to_randomize = [0, 1, 2, 3, 4]
traj_obj = ThreeDCircularTraj_fixed(radius=np.array([2, 1, 0]),
                                    freq=np.array([2, 2, 0]),
                                    yaw_bool=False)

high_level_folder = "priors/quad_geom_sysid_data/"

gather_batched_quadrotor_sysid_data(n_datapoints,
                                    batch_size,
                                    num_batches_per_intrinsic,
                                    gains_to_randomize,
                                    thetas_to_randomize,
                                    high_level_folder,
                                    randomize_trajs=False,
                                    traj_obj=traj_obj)
