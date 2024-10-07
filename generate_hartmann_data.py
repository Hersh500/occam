from learned_ctrlr_opt.systems.hartmann import gather_theta_batched_hartmann_data, HartmannFnParamsTrain

num_batches = 100
batch_size = 32
n_datapoints = num_batches * batch_size
thetas_to_randomize = [0, 1, 2, 3]
gains_to_randomize = [0, 1, 2, 3, 4, 5]

gather_theta_batched_hartmann_data(n_datapoints,
                                   batch_size,
                                   gains_to_randomize,
                                   thetas_to_randomize,
                                   "priors/hartmann/",
                                   params_t=HartmannFnParamsTrain)
