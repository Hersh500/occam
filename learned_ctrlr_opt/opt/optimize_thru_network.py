import numpy as np
import torch
import torch.nn
from torch.autograd import Variable

# Assumes network inputs are [0, 1]
def optimize_network_inputs(eval_fn,
                            input_dim,
                            num_ga_steps,
                            step_size,
                            cost_weights,
                            device,
                            fixed_inputs=None,
                            batch_size=64):
    inputs_to_optimize = Variable(torch.from_numpy(np.random.rand(batch_size, input_dim)))
    inputs_to_optimize.requires_grad = True
    optimizer = torch.optim.Adam([inputs_to_optimize], lr=step_size)
    if fixed_inputs is not None:
        fixed_inputs_batch = fixed_inputs.repeat(batch_size).reshape((batch_size, fixed_inputs.size(-1)))
    cost_weights_batch = torch.from_numpy(cost_weights).repeat(batch_size).reshape((batch_size, cost_weights.shape[-1]))
    cost_weights_batch = cost_weights_batch.to(device)
    for step in range(num_ga_steps):
        optimizer.zero_grad()
        if fixed_inputs is not None:
            full_inputs = torch.cat([inputs_to_optimize, fixed_inputs_batch], dim=-1).float().to(device)
        else:
            full_inputs = inputs_to_optimize.float().to(device)
        ys = eval_fn(full_inputs) * cost_weights_batch
        losses = -1 * torch.sum(ys)  # can I just sum everything up? gradients should be fine.
        losses.backward()
        optimizer.step()

    final_ys = torch.sum(ys, dim=-1)
    best_input_idx = torch.argmin(final_ys)
    # optimizer.zero_grad()
    return inputs_to_optimize[best_input_idx], best_input_idx, inputs_to_optimize, final_ys
