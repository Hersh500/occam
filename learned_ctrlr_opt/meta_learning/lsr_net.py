import torch
from torch import nn
import numpy as np
from learned_ctrlr_opt.meta_learning.reptile_net import ReptileModel, reptile_inner_opt
from torch.autograd import Variable
from learned_ctrlr_opt.utils.learning_utils import create_network


# Network that outputs a matrix of basis functions, with a least squares fitting as the last layer
class LSRBasisNet(nn.Module):
    def __init__(self,
                 in_size,
                 n_bases,
                 n_targets,
                 layer_sizes=[256, 256, 256],
                 nonlin="relu"):
        super().__init__()
        self.in_size = in_size
        self.n_bases = n_bases
        self.n_targets = n_targets
        self.use_last_layer = True
        self.nonlin_str = nonlin

        if nonlin.lower() == "relu":
            self.nonlin = nn.ReLU
        elif nonlin.lower() == "sigmoid":
            self.nonlin = nn.Sigmoid
        else:
            raise NotImplementedError()
        self.net = nn.Sequential(nn.Linear(in_size, layer_sizes[0]), self.nonlin())
        self.layer_sizes = layer_sizes
        for i, s in enumerate(layer_sizes[:-1]):
            self.net.append(nn.Linear(s, layer_sizes[i + 1]))
            self.net.append(self.nonlin())
        self.net.append(nn.Linear(layer_sizes[-1], self.n_bases * self.n_targets))
        self.last_layer = nn.Parameter(torch.zeros(self.n_bases), requires_grad=True)
        # MAP parameters -- unused except in KF updates
        # should be zero or randn?
        self.last_layer_prior = nn.Parameter(torch.zeros(self.n_bases), requires_grad=True)

        # need to maintain positive semidefinite constraints on these guys, which is not enforced during GD.
        # Instead, keep them as sqrts and in basis_kf, do XX.T to get the full matrices
        self.last_layer_prior_cov_sqrt = nn.Parameter(torch.zeros(self.n_bases, self.n_bases), requires_grad=True)
        self.Q_sqrt = nn.Parameter(torch.zeros(self.n_bases, self.n_bases), requires_grad=True)
        self.R_sqrt = nn.Parameter(torch.zeros(self.n_targets, self.n_targets), requires_grad=True)
        self.regularization_prior = nn.Parameter(torch.ones(1) * 1e-1)

    def forward(self, x):
        # currently doesn't support more than one batch dimension
        out_flat = self.net(x)
        out_rs = torch.reshape(out_flat, (x.size(0), self.n_bases, self.n_targets))
        if self.use_last_layer:
            tmp = torch.sum(out_rs * self.last_layer.unsqueeze(0).unsqueeze(-1), dim=-2)
            return tmp
        else:
            return out_rs

    def point_grad_to(self, target):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                if self.is_cuda():
                    p.grad = Variable(torch.zeros(p.size())).cuda()
                else:
                    p.grad = Variable(torch.zeros(p.size()))
            p.grad.data.zero_()  # not sure this is required
            p.grad.data.add_(p.data - target_p.data)

    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def clone(self):
        cloned_net = LSRBasisNet(self.in_size,
                                 self.n_bases,
                                 self.n_targets,
                                 self.layer_sizes,
                                 self.nonlin_str)
        cloned_net.load_state_dict(self.state_dict())
        if self.is_cuda():
            cloned_net.cuda()
        return cloned_net

    def initialize_priors(self):
        self.last_layer_prior = nn.Parameter(torch.clone(self.last_layer).detach(), requires_grad=True)
        self.last_layer_prior_cov_sqrt = nn.Parameter(torch.eye(self.n_bases), requires_grad=True)
        self.Q_sqrt = nn.Parameter(torch.eye(self.n_bases), requires_grad=True)
        self.R_sqrt = nn.Parameter(torch.eye(self.n_targets), requires_grad=True)
        self.regularization_prior = nn.Parameter(torch.ones(1) * 1e-1, requires_grad=True)


class LSRBasisNet_encoder(nn.Module):
    def __init__(self,
                 gain_in_size,
                 n_bases,
                 n_targets,
                 history_in_size,
                 history_out_size,
                 encoder_layer_sizes,
                 encoder_nonlin,
                 layer_sizes=[256, 256, 256],
                 nonlin="relu"):

        super().__init__()
        self.in_size = gain_in_size
        self.n_bases = n_bases
        self.n_targets = n_targets
        self.use_last_layer = True
        self.history_in_size = history_in_size
        self.history_out_size = history_out_size
        self.encoder_layer_sizes = encoder_layer_sizes
        self.encoder_nonlin = encoder_nonlin
        self.encoder_network = create_network(history_in_size, history_out_size,
                                             encoder_layer_sizes, encoder_nonlin)
        self.nonlin_str = nonlin

        if nonlin.lower() == "relu":
            self.nonlin = nn.ReLU
        elif nonlin.lower() == "sigmoid":
            self.nonlin = nn.Sigmoid
        else:
            raise NotImplementedError()
        # TODO(hersh500): replace this with create_network()
        self.net = nn.Sequential(nn.Linear(gain_in_size + history_out_size, layer_sizes[0]), self.nonlin())
        self.layer_sizes = layer_sizes
        for i, s in enumerate(layer_sizes[:-1]):
            self.net.append(nn.Linear(s, layer_sizes[i + 1]))
            self.net.append(self.nonlin())
        self.net.append(nn.Linear(layer_sizes[-1], self.n_bases * self.n_targets))
        self.last_layer = nn.Parameter(torch.zeros(self.n_bases), requires_grad=True)
        self.regularization_prior = nn.Parameter(torch.ones(1) * 1e-1)

        # MAP parameters -- unused except in KF updates
        # should be zero or randn?
        self.last_layer_prior = nn.Parameter(torch.zeros(self.n_bases), requires_grad=True)

        # need to maintain positive semidefinite constraints on these guys, which is not enforced during GD.
        # Instead, keep them as sqrts and in basis_kf, do XX.T to get the full matrices
        self.last_layer_prior_cov_sqrt = nn.Parameter(torch.zeros(self.n_bases, self.n_bases), requires_grad=True)
        self.Q_sqrt = nn.Parameter(torch.zeros(self.n_bases, self.n_bases), requires_grad=True)
        self.R_sqrt = nn.Parameter(torch.zeros(self.n_targets, self.n_targets), requires_grad=True)

    def forward(self, x):
        history = x[..., self.in_size:]
        g = x[..., :self.in_size]
        history_out = self.encoder_network(history)
        net_in = torch.cat([g, history_out], dim=-1)
        out_flat = self.net(net_in)
        out_rs = torch.reshape(out_flat, (g.size(0), self.n_bases, self.n_targets))
        if self.use_last_layer:
            tmp = torch.sum(out_rs * self.last_layer.unsqueeze(0).unsqueeze(-1), dim=-2)
            return tmp
        else:
            return out_rs
        
    def point_grad_to(self, target):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                if self.is_cuda():
                    p.grad = Variable(torch.zeros(p.size())).cuda()
                else:
                    p.grad = Variable(torch.zeros(p.size()))
            p.grad.data.zero_()  # not sure this is required
            p.grad.data.add_(p.data - target_p.data)

    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def clone(self):
        cloned_net = LSRBasisNet_encoder(self.in_size,
                                         self.n_bases,
                                         self.n_targets,
                                         self.history_in_size,
                                         self.history_out_size,
                                         self.encoder_layer_sizes,
                                         self.encoder_nonlin,
                                         self.layer_sizes,
                                         self.nonlin_str)
        cloned_net.load_state_dict(self.state_dict())
        if self.is_cuda():
            cloned_net.cuda()
        return cloned_net

    def initialize_priors(self):
        self.last_layer_prior = nn.Parameter(torch.clone(self.last_layer).detach(), requires_grad=True)
        self.last_layer_prior_cov_sqrt = nn.Parameter(torch.eye(self.n_bases), requires_grad=True)
        self.Q_sqrt = nn.Parameter(torch.eye(self.n_bases), requires_grad=True)
        self.R_sqrt = nn.Parameter(torch.eye(self.n_targets), requires_grad=True)
        self.regularization_prior = nn.Parameter(torch.ones(1) * 1e-1, requires_grad=True)


# A: (N, N_b, N_y)
# y: (N, N_y)
# returns w: (N_b, 1)
def intrinsic_batched_lstsq(A, y_gt, y_noisy=None, random_sample_num=False, use_half=False, l=0.1):
    assert A.size(0) == y_gt.size(0)
    assert A.size(-1) == y_gt.size(-1)
    assert A.device == y_gt.device

    if random_sample_num:  # train LSF on subset of points, use rest for residual
        num_to_use = np.random.randint(1, A.size(0))
        #         num_to_use = 32
        idxs_to_use = np.random.choice(A.size(0), num_to_use, replace=False)
    elif use_half:  # train LSF and use train loss for residual
        num_to_use = int(A.size(0) / 2)
        idxs_to_use = np.arange(num_to_use)
    else:
        num_to_use = A.size(0)
        idxs_to_use = np.arange(num_to_use)

    # A_all = torch.zeros((A.size(0), A.size(1)))
    A_all = torch.zeros((A.size(1), A.size(1))).to(A.device)
    target_all = torch.zeros(A.size(1), 1).to(A.device)
    if y_noisy is None:
        y_noisy = y_gt
    for i in range(y_noisy.shape[-1]):
        A_all += torch.mm(torch.t(A[idxs_to_use, :, i]), A[idxs_to_use, :, i])
        target_all += torch.matmul(torch.t(A[idxs_to_use, :, i]), y_noisy[idxs_to_use, i]).unsqueeze(-1)
    # ridge regression
    result = torch.linalg.lstsq(A_all + torch.eye(A.size(1)).float().to(A.device) * l, target_all)
    # sometimes pytorch does not compute the residuals, which is necessary to compute the gradients for the network.
    # predicted_y = torch.zeros(y.size()).to(A.device)
    predicted_y = torch.zeros(A.size(0), *y_noisy.size()[1:]).to(A.device)
    for j in range(A.size(1)):
        predicted_y += A[:, j, :] * result.solution[j]
    criterion = torch.nn.MSELoss()
    residual = criterion(predicted_y, y_gt)

    unseen_loss = 0
    if num_to_use < A.size(0):
        pred_y_unseen = torch.zeros(A.size(0) - num_to_use, *y_noisy.size()[1:]).to(A.device)
        unseen_idxs = [k for k in range(A.size(0)) if k not in idxs_to_use]
        for j in range(A.size(1)):
            pred_y_unseen += A[unseen_idxs, j, :] * result.solution[j]
        unseen_loss = criterion(pred_y_unseen, y_gt[unseen_idxs, ...])
    return result.solution, residual, predicted_y, unseen_loss


def pretrain_basis_net_no_adapt(train_dataloader,
                                basis_net,
                                optimizer,
                                input_batch_idxs,
                                target_batch_idx_gt,
                                device,
                                target_batch_idx_noisy=None):
    avg_train_residual = 0
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.L1Loss()
    for j, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        batches = data
        input_batches = []
        for idx in input_batch_idxs:
            input_batches.append(batches[idx])
        input_batch = torch.cat(input_batches, dim=-1)
        output = basis_net(input_batch.float().to(device))
        metric_gt_batch = batches[target_batch_idx_gt]
        loss = criterion(output, metric_gt_batch.float().to(device))
        loss.backward()
        optimizer.step()
        avg_train_residual += loss
    return avg_train_residual / len(train_dataloader)


def eval_basis_net_no_adapt(val_dataloader,
                            basis_net,
                            input_batch_idxs,
                            target_batch_idx_gt,
                            device,
                            target_batch_idx_noisy=None):
    avg_val_residual = 0
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.L1Loss()
    for j, data in enumerate(val_dataloader):
        batches = data
        input_batches = []
        for idx in input_batch_idxs:
            input_batches.append(batches[idx])
        input_batch = torch.cat(input_batches, dim=-1)
        with torch.no_grad():
            output = basis_net(input_batch.float().to(device))
            metric_gt_batch = batches[target_batch_idx_gt]
            loss = criterion(output, metric_gt_batch.float().to(device))
            avg_val_residual += loss
    return avg_val_residual / len(val_dataloader)


def pretrain_basis_net_reptile(train_dataloader,
                               basis_net,
                               meta_optimizer,
                               inner_lr,
                               num_inner_steps,
                               input_batch_idxs,
                               target_batch_idx_gt,
                               random_num_lstsq,
                               device,
                               num=None,
                               target_batch_idx_noisy=None):
    assert random_num_lstsq or (num is not None)
    meta_optimizer.zero_grad()  # when to zero out gradients?
    avg_train_residual = 0
    for i, task_batch in enumerate(train_dataloader):
        inner_net = basis_net.clone()
        inner_optimizer = torch.optim.Adam(inner_net.parameters(), lr=inner_lr)
        inner_loss = torch.nn.MSELoss()
        batches = task_batch
        input_batches = []
        for idx in input_batch_idxs:
            input_batches.append(batches[idx])
        input_batch = torch.cat(input_batches, dim=-1)
        gt_batch = batches[target_batch_idx_gt]
        n = np.random.randint(1, input_batch.size(0) - 1) if random_num_lstsq else num
        inner_net, final_loss, unseen_loss = reptile_inner_opt(inner_net,
                                                               inner_optimizer,
                                                               inner_loss,
                                                               input_batch.float().to(device),
                                                               gt_batch.float().cuda(),
                                                               K=n,
                                                               num_steps=num_inner_steps)

        avg_train_residual += final_loss
        basis_net.point_grad_to(inner_net)
        meta_optimizer.step()
    avg_train_residual /= len(train_dataloader)
    return avg_train_residual


def eval_basis_net_reptile(val_dataloader,
                           basis_net,
                           inner_lr,
                           num_inner_steps,
                           input_batch_idxs,
                           target_batch_idx_gt,
                           device,
                           num):
    avg_val_residual = 0
    avg_val_residual_unseen = 0
    for i, task_batch in enumerate(val_dataloader):
        inner_net = basis_net.clone()
        inner_optimizer = torch.optim.Adam(inner_net.parameters(), lr=inner_lr)
        inner_loss = torch.nn.MSELoss()
        batches = task_batch
        input_batches = []
        for idx in input_batch_idxs:
            input_batches.append(batches[idx])
        input_batch = torch.cat(input_batches, dim=-1)
        gt_batch = batches[target_batch_idx_gt]
        inner_net, final_loss, unseen_loss = reptile_inner_opt(inner_net,
                                                               inner_optimizer,
                                                               inner_loss,
                                                               input_batch.float().to(device),
                                                               gt_batch.float().cuda(),
                                                               K=num,
                                                               num_steps=num_inner_steps,
                                                               random=False)
        avg_val_residual += final_loss
        avg_val_residual_unseen += unseen_loss

    avg_val_residual /= len(val_dataloader)
    avg_val_residual_unseen /= len(val_dataloader)
    return avg_val_residual, avg_val_residual_unseen


def train_basis_net_lstsq(train_dataloader,
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
        gt_batch = batches[target_batch_idx_gt]
        noisy_batch = None if target_batch_idx_noisy is None else batches[target_batch_idx_noisy].float().to(device)
        soln, loss, _, unseen_loss = intrinsic_batched_lstsq(output,
                                                             y_noisy=noisy_batch,
                                                             random_sample_num=random_num_lstsq,
                                                             y_gt=gt_batch.float().to(device),
                                                             use_half=True,
                                                             l=basis_net.regularization_prior[0])
        loss.backward()
        optimizer.step()
        avg_train_residual += loss
    return avg_train_residual / len(train_dataloader)


def eval_basis_net_lstsq(val_dataloader,
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
        gt_batch = batches[target_batch_idx_gt]
        noisy_batch = None if target_batch_idx_noisy is None else batches[target_batch_idx_noisy].float().to(device)
        with torch.no_grad():
            output = basis_net(input_batch.float().to(device))
        soln, loss, _, unseen_loss = intrinsic_batched_lstsq(output,
                                                             y_noisy=noisy_batch,
                                                             random_sample_num=False,
                                                             y_gt=gt_batch.float().to(device),
                                                             use_half=True,
                                                             l=basis_net.regularization_prior[0])
        # loss = criterion(loss, metric_batch.float().to(device))
        avg_val_residual += loss
        avg_val_residual_unseen += unseen_loss
    return avg_val_residual / len(val_dataloader), avg_val_residual_unseen / len(val_dataloader)

def last_layer_prediction(x, network, weights):
    device = next(network.parameters()).device
    bases = network(x.float().to(device))
    predicted_y = torch.zeros(bases.size(0), bases.size(-1)).to(device)
    for j in range(bases.size(1)):
        predicted_y += bases[:, j, :] * weights[j]
    return predicted_y


def calc_weights_dumb(x_data, y_data, network):
    device = next(network.parameters()).device
    bases = network(x_data.float().to(device))
    with torch.no_grad():
        weights, residual, pred_y, unseen_res = intrinsic_batched_lstsq(bases, y_data.float().to(device), l=network.regularization_prior[0])
    return weights
