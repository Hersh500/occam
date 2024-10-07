import numpy as np
import torch
from torch.autograd import Variable
from bo_wrappers.botorch_utils import initialize_model
from botorch.fit import fit_gpytorch_mll


def update_theta_mle(prior_model, new_x, new_y, theta_hat, num_steps=30, step_size=0.0001):
    theta_size = theta_hat.shape[0]
    num_points = new_x.shape[0]
    prior_model.eval()
    for i in range(num_steps):
        theta_exp = torch.from_numpy(theta_hat).expand(num_points, theta_size)
        # There's probably a way to streamline this...
        x_all_torch = Variable(torch.hstack([torch.from_numpy(new_x), theta_exp]), requires_grad=True)
        y_torch = torch.from_numpy(new_y)
        log_probs = prior_model(x_all_torch).log_prob(torch.unsqueeze(y_torch, 2))
        # print(f"grad before backward call= {x_all_torch.grad[:,-theta_size:]}")
        torch.sum(log_probs).backward()
        theta_grad = np.mean(x_all_torch.grad[:,-theta_size:].detach().numpy(), axis=0)
        theta_hat -= step_size * theta_grad
    return theta_hat


def predict(theta_particles_norm, explore_noise):
    new_particles = theta_particles_norm + np.random.randn(*theta_particles_norm.shape) * explore_noise
    return np.clip(new_particles, 0, 1)


# Simply resamples (with replacement) the particles proportionally to their weights
def resample_particles_simple(theta_particles, normalized_weights, num=None):
    num_particles = theta_particles.shape[0]
    if num is None:
        num = num_particles
    indices = np.random.choice(num_particles, num, p=normalized_weights, replace=True)
    new_weights = np.ones(num) * 1/num
    return theta_particles[indices], new_weights


def update_weights(theta_particles_norm,
                   obs_norm,
                   prior,
                   x_eval_norm,
                   weights,
                   device):
    num_gains = x_eval_norm.shape[0]
    x_all_norm = np.zeros((theta_particles_norm.shape[0], num_gains+theta_particles_norm.shape[1]))
    x_all_norm[:,:num_gains] = x_eval_norm
    x_all_norm[:,num_gains:] = theta_particles_norm
    # x_all_norm = normalize(x_all, all_bounds)
    x_all_torch = torch.unsqueeze(torch.from_numpy(x_all_norm), 0)
    new_weights = np.zeros(weights.shape)
    for p in range(theta_particles_norm.shape[0]):
        distr = prior(x_all_torch[:,p,:].to(device))
        likelihood = torch.sum(distr.log_prob(torch.unsqueeze(torch.from_numpy(obs_norm).to(device), 1)), dim=1)
        # print(f"likelihood = {likelihood}")
        new_weights[p] = weights[p] * np.exp(likelihood.cpu().detach().numpy())
    new_weights /= np.sum(new_weights)
    # print(f"new_weights = {new_weights}")
    return new_weights


def prune_particles(theta_particles_norm, weights, num=20):
    indices_sorted = np.argsort(weights)[::-1][:num]
    return theta_particles_norm[indices_sorted], weights[indices_sorted]


def compute_ranking_loss(preds, targets, debug=False):
    sum = 0
    num_tasks = targets.shape[-1]
    # print(f"preds shape is {preds.shape}")
    for i in range(0, preds.shape[0]):
        for j in range(i+1, preds.shape[0]):
            for k in range(num_tasks):
                if debug:
                    print("---------------------------")
                    print(f"Preds[i] = {preds[i][k]}, targets[i] = {targets[i][k]}")
                    print(f"Preds[j] = {preds[j][k]}, targets[j] = {targets[j][k]}")
                sum += int((preds[i][k] <= preds[j][k]) != (targets[i][k] <= targets[j][k]))
    return sum


def compute_target_model_loss(x_eval_norm, y_eval_norm):
    num_points = x_eval_norm.shape[0]
    num_gains = x_eval_norm.shape[1]
    loss = 0
    state_dict = None
    preds = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for i in range(num_points):
        x_rs = x_eval_norm.reshape(1, num_points, x_eval_norm.shape[1])
        y_rs = y_eval_norm.reshape(1, num_points, y_eval_norm.shape[1])
        point_out = x_eval_norm[i]
        mask = np.ones((1, num_points), dtype=bool)
        mask[0,i] = False
        x = torch.from_numpy(x_rs[mask]).double()
        y = torch.from_numpy(y_rs[mask]).double()
        current_mll, current_model = initialize_model(x.to(device), y.to(device))
        fit_gpytorch_mll(current_mll)
        if state_dict is None:
            state_dict = current_model.state_dict()
        pred_distr = current_model(torch.from_numpy(point_out).view(1, 1, num_gains).double().to(device))
        pred = pred_distr.mean.cpu().detach().numpy()
        preds.append(pred)
    preds = np.array(preds)[:,:,0]
    # return compute_ranking_loss(preds @ np.reshape(cost_weights, (-1, 1)), y_sc, debug=False)
    return compute_ranking_loss(preds, y_eval_norm, debug=False)


# Instead of using the raw values/likelihood, this uses the RGPE approach
# of giving higher weight to the priors which correctly order the evaluated points
# (ie. they are able to predict the optimum)
# Prior is static; could cache predictions on previous evals to speed it up?
def update_weights_ordering(theta_particles_norm,
                            obs_norm_all,
                            prior,
                            x_eval_norm_all,
                            device,
                            cached_prev_preds=None):
    num_gains = x_eval_norm_all.shape[1]
    if cached_prev_preds is None:
        predicted_targets = np.zeros((x_eval_norm_all.shape[0], theta_particles_norm.shape[0], obs_norm_all.shape[-1]))
        for i in range(x_eval_norm_all.shape[0]):
            # Have to repeat this step for each theta particle
            x_norm = np.zeros((theta_particles_norm.shape[0], num_gains+theta_particles_norm.shape[1]))
            x_norm[:,:num_gains] = x_eval_norm_all[i,:]
            x_norm[:,num_gains:] = theta_particles_norm
            x_torch = torch.unsqueeze(torch.from_numpy(x_norm), 0)
            distr = prior(x_torch.to(device))
            for p in range(theta_particles_norm.shape[0]):
                predicted_targets[i, p] = distr.mean.cpu().detach().numpy()[0, :, p]
    elif cached_prev_preds.shape[0] != obs_norm_all.shape[0] - 1:
        raise Exception(f"cached prev preds has wrong 0 shape: {cached_prev_preds.shape[0]}, should be {obs_norm_all.shape[0] - 1}")
    else:
        predicted_target = np.zeros(1, theta_particles_norm.shape[0], obs_norm_all.shape[-1])
        x_norm = np.zeros((theta_particles_norm.shape[0], num_gains + theta_particles_norm.shape[1]))
        x_norm[:, :num_gains] = x_eval_norm_all[-1, :]
        x_norm[:, num_gains:] = theta_particles_norm
        x_torch = torch.unsqueeze(torch.from_numpy(x_norm), 0)
        distr = prior(x_torch.to(device))
        for p in range(theta_particles_norm.shape[0]):
            predicted_target[0, p] = distr.mean.cpu().detach().numpy()[0, :, p]
        predicted_targets = np.concatenate([cached_prev_preds, predicted_target], axis=0)
        print(f"predicted targets shape is {predicted_targets.shape}")
    losses = np.zeros(theta_particles_norm.shape[0])
    for i in range(theta_particles_norm.shape[0]):
        losses[i] = compute_ranking_loss(predicted_targets[:,i], obs_norm_all)
    target_loss = compute_target_model_loss(x_eval_norm_all, obs_norm_all)
    print(f"Target Loss is {target_loss}")
    print(f"prior losses are {losses}")
    all_losses = np.append(losses, target_loss)
    all_losses = np.max(all_losses) - all_losses
    all_losses /= (np.sum(all_losses))
    # print(f"new weights are {losses}")
    # print(f"sum of losses is {np.sum(all_losses)}")
    return all_losses[:-1], all_losses[-1], predicted_targets
