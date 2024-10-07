import numpy as np
import torch


def kalman_step(w, sigma, z, phi, Q, R):
    w_pred = w
    sigma_pred = sigma + Q

    y = z - torch.matmul(torch.t(phi), w_pred)
    h_cov = torch.mm(torch.t(phi), torch.mm(sigma_pred, phi)) + R
    K = torch.mm(torch.mm(sigma_pred, phi), torch.linalg.inv(h_cov))
    w_post = w_pred + torch.matmul(K, y).squeeze()
    sigma_post = torch.mm(torch.eye(w.size(0)).float().to(w.device) - torch.mm(K, torch.t(phi)), sigma_pred)
    return w_post, sigma_post, K


def kf_training(A,
                y_gt,
                prior_mean,
                prior_cov_sqrt,
                Q_sqrt,
                R_sqrt,
                random_sample_num=False,
                use_half=False,
                shuffle=True):
    assert A.size(0) == y_gt.size(0)
    assert A.size(-1) == y_gt.size(-1)
    assert A.device == y_gt.device

    if random_sample_num:  # train LSF on subset of points, use rest for residual
        num_to_use = np.random.randint(1, A.size(0))
        idxs_to_use = np.random.choice(A.size(0), num_to_use, replace=False)
    elif use_half:  # train LSF and use train loss for residual
        num_to_use = int(A.size(0) / 2)
        idxs_to_use = np.arange(num_to_use)
    else:
        num_to_use = A.size(0)
        idxs_to_use = np.arange(num_to_use)

    if shuffle:
        np.random.shuffle(idxs_to_use)

    prior_cov = torch.mm(prior_cov_sqrt, torch.t(prior_cov_sqrt))
    Q = torch.mm(Q_sqrt, torch.t(Q_sqrt))
    R = torch.mm(R_sqrt, torch.t(R_sqrt))

    kalman_gains = torch.zeros(num_to_use, A.size(1), A.size(-1))
    weights = torch.zeros(num_to_use, A.size(1))
    covs = torch.zeros(num_to_use, A.size(1), A.size(1))
    w = prior_mean
    sigma = prior_cov
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.L1Loss()
    for t, data_idx in enumerate(idxs_to_use):
        w, sigma, K = kalman_step(w,
                                  sigma,
                                  y_gt[data_idx],
                                  A[data_idx],
                                  Q,
                                  R)
        kalman_gains[t, :] = K
        weights[t] = w
        covs[t] = sigma

    final_weights = weights[-1]

    # Proxy condition for validation batch
    predicted_y_seen = torch.zeros(num_to_use, *y_gt.size()[1:]).to(A.device)
    for j in range(A.size(1)):
        predicted_y_seen += A[idxs_to_use, j, :] * final_weights[j]
    seen_loss = criterion(predicted_y_seen, y_gt[idxs_to_use, ...])

    predicted_y = torch.zeros(A.size(0), *y_gt.size()[1:]).to(A.device)
    for j in range(A.size(1)):
        predicted_y += A[:, j, :] * final_weights[j]
    full_loss = criterion(predicted_y, y_gt)

    unseen_loss = 0
    if num_to_use < A.size(0):
        pred_y_unseen = torch.zeros(A.size(0) - num_to_use, *y_gt.size()[1:]).to(A.device)
        unseen_idxs = [k for k in range(A.size(0)) if k not in idxs_to_use]
        for j in range(A.size(1)):
            pred_y_unseen += A[unseen_idxs, j, :] * final_weights[j]
        unseen_loss = criterion(pred_y_unseen, y_gt[unseen_idxs, ...])
    # print(f"Loss on Seen points (fitting error) was {seen_loss}, unseen was {unseen_loss}")
    return final_weights, full_loss, unseen_loss, weights, covs, kalman_gains


def train_basis_net_kf(train_dataloader,
                       basis_net,
                       optimizer,
                       input_batch_idxs,
                       target_batch_idx_gt,
                       random_num_lstsq,
                       device,
                       target_batch_idx_noisy=None):
    avg_train_residual = 0
    for j, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        batches = data
        input_batches = []
        for idx in input_batch_idxs:
            input_batches.append(batches[idx])
        input_batch = torch.cat(input_batches, dim=-1)
        output = basis_net(input_batch.float().to(device))
        gt_batch = batches[target_batch_idx_gt].float().to(device)
        noisy_batch = None if target_batch_idx_noisy is None else batches[target_batch_idx_noisy].float().to(device)
        final_weights, full_loss, unseen_loss, weights, covs, kalmans = kf_training(output,
                                                                                    gt_batch,
                                                                                    basis_net.last_layer_prior,
                                                                                    basis_net.last_layer_prior_cov_sqrt,
                                                                                    basis_net.Q_sqrt,
                                                                                    basis_net.R_sqrt,
                                                                                    random_sample_num=random_num_lstsq,
                                                                                    use_half=True,
                                                                                    # random num take precendence
                                                                                    shuffle=True)
        full_loss.backward()
        optimizer.step()
        avg_train_residual += full_loss
    return avg_train_residual / len(train_dataloader)


def eval_basis_net_kf(val_dataloader,
                      basis_net,
                      input_batch_idxs,
                      target_batch_idx_gt,
                      device,
                      target_batch_idx_noisy=None):

    avg_val_residual = 0
    avg_val_residual_unseen = 0
    for j, data in enumerate(val_dataloader):
        batches = data
        input_batches = []
        for idx in input_batch_idxs:
            input_batches.append(batches[idx])
        input_batch = torch.cat(input_batches, dim=-1)
        gt_batch = batches[target_batch_idx_gt].float().to(device)
        noisy_batch = None if target_batch_idx_noisy is None else batches[target_batch_idx_noisy].float().to(device)
        with torch.no_grad():
            output = basis_net(input_batch.float().to(device))
            final_weights, full_loss, unseen_loss, weights, covs, kalmans = kf_training(output,
                                                                                        gt_batch,
                                                                                        basis_net.last_layer_prior,
                                                                                        basis_net.last_layer_prior_cov_sqrt,
                                                                                        basis_net.Q_sqrt,
                                                                                        basis_net.R_sqrt,
                                                                                        use_half=True,
                                                                                        # random num take precendence
                                                                                        shuffle=True)

        avg_val_residual += full_loss
        avg_val_residual_unseen += unseen_loss
    return avg_val_residual / len(val_dataloader), avg_val_residual_unseen / len(val_dataloader)


# x: (N, n_b, n_y)
# sigma: (n_b, n_b)
# weights: (n_b, 1)
# Only gets used in evaluation settings
def last_layer_prediction_uncertainty_aware(x, network, weights_mean, sigma):
    device = next(network.parameters()).device
    with torch.no_grad():
        bases = network(x.float().to(device))
    predicted_y_mean = torch.zeros(bases.size(0), bases.size(-1), device=device)
    predicted_y_sigma = torch.zeros(bases.size(0), bases.size(-1), bases.size(-1), device=device)
    for j in range(bases.size(0)):
        # predicted_y_mean += bases[:, j, :] * weights_mean[j]
        predicted_y_mean[j] = torch.matmul(torch.t(bases[j]), weights_mean)
        predicted_y_sigma[j] = torch.matmul(torch.t(bases[j]), torch.matmul(sigma, bases[j]))
    return predicted_y_mean, predicted_y_sigma
