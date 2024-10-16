import numpy as np
from learned_ctrlr_opt.systems.upkie.upkie import gather_upkie_balancing_mpc_data

initial_state_bounds = np.array([[0, 0],
                                 [-0.04, 0.04],
                                 [-0.15, 0.15],
                                 [-0.02, 0.02]])

num_batches = 500
batch_size = 32
thetas_to_randomize = [0, 1, 3]
ep_length = 3

gather_upkie_balancing_mpc_data(num_batches,
                                batch_size,
                                thetas_to_randomize,
                                "priors/upkie",
                                initial_state_bounds,
                                ep_length)
