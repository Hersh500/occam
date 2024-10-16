import numpy as np
from learned_ctrlr_opt.meta_learning.gather_sysid_data import gather_batched_quad_geom_rss_sysid_data
from learned_ctrlr_opt.systems.quadrotor_geom import ThreeDCircularTraj_fixed, QuadrotorSE3Control_RandomStart, CrazyFlieSE3Control_RandomStart

batch_size = 64
num_batches_per_intrinsic = 1
n_datapoints = batch_size * 100
# n_datapoints = batch_size * 5
gains_to_randomize = [0, 1, 2, 3, 4, 5, 6, 7]
thetas_to_randomize = [0, 1, 2, 3, 4]
initial_tf = 2
initial_sensor_traj_length_tsteps = 50
traj_obj = ThreeDCircularTraj_fixed(radius=np.array([1, 1, 0.5]),
                                    freq=np.array([0.3, 0.3, 0.3]),
                                    yaw_bool=False)

high_level_folder = "priors/quad_geom_rss_sysid_data/crazyflie/"

gather_batched_quad_geom_rss_sysid_data(n_datapoints,
                                        batch_size,
                                        num_batches_per_intrinsic,
                                        gains_to_randomize,
                                        thetas_to_randomize,
                                        high_level_folder,
                                        initial_tf,
                                        initial_sensor_traj_length_tsteps,
                                        randomize_tracks=True,
                                        same_track_batches=False,
                                        traj_obj=traj_obj,
                                        t_f=4,
                                        quad_t=CrazyFlieSE3Control_RandomStart,
                                        test_set=True)
