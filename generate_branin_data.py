from learned_ctrlr_opt.systems.branin import gather_theta_batched_branin_data, BraninFnParamsTrain

num_batches = 80
batch_size = 32
n_datapoints = num_batches * batch_size
thetas_to_randomize = [0, 1, 2, 3, 4, 5]
gains_to_randomize = [0, 1]

gather_theta_batched_branin_data(n_datapoints, 
                                 batch_size, 
                                 gains_to_randomize,
                                 thetas_to_randomize,
                                 "priors/branin/",
                                 params_t=BraninFnParamsTrain)
