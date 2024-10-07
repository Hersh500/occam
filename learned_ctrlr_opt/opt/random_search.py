import numpy as np
import torch
from learned_ctrlr_opt.utils.dataset_utils import denormalize


def random_search_simple(num_samples, eval_fn, in_dim, bounds, minimize=True):
    gains = denormalize(np.random.rand(num_samples, in_dim), bounds)
    acq_cost, cost = eval_fn(gains)
    best_gain = np.zeros(in_dim)
    if minimize:
        best_acq_cost = np.inf
        best_surr_cost = np.inf
    else:
        best_acq_cost = -np.inf
        best_surr_cost = -np.inf
    for i in range(num_samples):
        if minimize:
            if acq_cost[i] < best_acq_cost:
                best_gain = gains[i]
                best_acq_cost = acq_cost[i]
                best_surr_cost = cost[i]
        else:
            if acq_cost[i] < best_acq_cost:
                best_gain = gains[i]
                best_acq_cost = acq_cost[i]
                best_curr_cost = cost[i]
    return best_gain, best_acq_cost, best_surr_cost


def random_search(eval_fn,
                  trial_budget,
                  cost_weights,
                  input_dim,
                  device,
                  fixed_inputs=None,
                  batch_size=64,
                  sigma_weight=None,
                  guaranteed_searches=None,
                  sampler=None):
    all_tried_gains = torch.zeros(trial_budget, input_dim)
    all_obtained_ys = torch.zeros(trial_budget, cost_weights.shape[-1])
    all_obtained_costs = torch.zeros(trial_budget)
    num_tried = 0
    current_best_cost = -10000
    current_best_gain = None
    current_best_y = 0
    fixed_inputs = fixed_inputs.to(device)
    # First evaluate the guaranteed searches
    if guaranteed_searches is not None and guaranteed_searches.shape[0] > 0:
        q = guaranteed_searches.shape[0]
        fixed_inputs_batch = fixed_inputs.repeat(q).reshape((q, fixed_inputs.size(-1)))
        full_inputs = torch.cat([guaranteed_searches.to(device), fixed_inputs_batch], dim=-1).float()

        losses, ys, vars = eval_fn(full_inputs, cost_weights, sigma_weight)
        best_in_batch = torch.argmax(losses)
        if losses[best_in_batch] > current_best_cost:
            current_best_cost = losses[best_in_batch]
            current_best_gain = guaranteed_searches[best_in_batch]
            current_best_y = np.dot(cost_weights, ys[best_in_batch].cpu().detach().numpy())
        all_tried_gains[num_tried:num_tried+q] = guaranteed_searches
        all_obtained_ys[num_tried:num_tried+q] = ys
        all_obtained_costs[num_tried:num_tried+q] = losses
        num_tried += q

    while num_tried < trial_budget:
        q = min(batch_size, trial_budget - num_tried)
        if sampler is None:
            inputs_to_try = torch.rand(q, input_dim, device=device)
        else:
            inputs_to_try = sampler(q)
        if fixed_inputs is not None:
            fixed_inputs_batch = fixed_inputs.repeat(q).reshape((q, fixed_inputs.size(-1)))
            full_inputs = torch.cat([inputs_to_try, fixed_inputs_batch], dim=-1).float()
        else:
            full_inputs = inputs_to_try.float()
        losses, ys, vars = eval_fn(full_inputs, cost_weights, sigma_weight)
        best_in_batch = torch.argmax(losses)
        if losses[best_in_batch] > current_best_cost:
            current_best_cost = losses[best_in_batch]
            current_best_gain = inputs_to_try[best_in_batch]
            current_best_y = np.dot(cost_weights, ys[best_in_batch].cpu().detach().numpy())
        all_tried_gains[num_tried:num_tried+q] = inputs_to_try
        all_obtained_ys[num_tried:num_tried+q] = ys
        all_obtained_costs[num_tried:num_tried+q] = losses
        num_tried += q
    return current_best_gain.detach().cpu(), current_best_cost, current_best_y, all_tried_gains, all_obtained_ys, all_obtained_costs
