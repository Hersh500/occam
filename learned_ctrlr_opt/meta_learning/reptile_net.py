import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from learned_ctrlr_opt.utils.learning_utils import torch_delete_batch_idxs, create_network

# https://github.com/gabrielhuang/reptile-pytorch/tree/master
class ReptileModel(nn.Module):

    def __init__(self, in_size, n_targets, layer_sizes=[256, 256, 256],
                 nonlin="relu"):
        super().__init__()
        self.in_size = in_size
        self.n_targets = n_targets
        self.nonlin_string = nonlin

        if nonlin.lower() == "relu":
            self.nonlin = nn.ReLU
        elif nonlin.lower() == "sigmoid":
            self.nonlin = nn.Sigmoid
        else:
            raise NotImplementedError()
        self.net = nn.Sequential(nn.Linear(in_size, layer_sizes[0]), self.nonlin())
        self.layer_sizes = layer_sizes
        for i, s in enumerate(layer_sizes[:-1]):
            self.net.append(nn.Linear(s, layer_sizes[i+1]))
            self.net.append(self.nonlin())
        self.net.append(nn.Linear(layer_sizes[-1], self.n_targets))


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

    def forward(self, x):
        return self.net(x)

    def clone(self):
        cloned_net = ReptileModel(self.in_size,
                                  self.n_targets,
                                  self.layer_sizes,
                                  self.nonlin_string)
        cloned_net.load_state_dict(self.state_dict())
        if self.is_cuda():
            cloned_net.cuda()
        return cloned_net

# some stickiness here with cloning the encoder network?
# might need to keep encoder net a low level
class ReptileModel_Encoder(nn.Module):
    def __init__(self, 
                 in_size, 
                 n_targets,
                 history_in_size,
                 history_out_size,
                 encoder_layer_sizes,
                 encoder_nonlin,
                 layer_sizes=[256, 256, 256],
                 nonlin="relu"):
        super().__init__()
        self.in_size = in_size
        self.n_targets = n_targets
        self.nonlin_string = nonlin
        self.history_in_size = history_in_size
        self.history_out_size = history_out_size
        self.encoder_layer_sizes = encoder_layer_sizes
        self.encoder_nonlin = encoder_nonlin
        self.encoder_network = create_network(history_in_size, history_out_size,
                                             encoder_layer_sizes, encoder_nonlin)


        if nonlin.lower() == "relu":
            self.nonlin = nn.ReLU
        elif nonlin.lower() == "sigmoid":
            self.nonlin = nn.Sigmoid
        else:
            raise NotImplementedError()
        self.net = nn.Sequential(nn.Linear(in_size+history_out_size, layer_sizes[0]), self.nonlin())
        self.layer_sizes = layer_sizes
        for i, s in enumerate(layer_sizes[:-1]):
            self.net.append(nn.Linear(s, layer_sizes[i+1]))
            self.net.append(self.nonlin())
        self.net.append(nn.Linear(layer_sizes[-1], self.n_targets))


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

    def forward(self, x):
        history = x[..., self.in_size:]
        g = x[..., :self.in_size]
        history_out = self.encoder_network(history)
        return self.net(torch.cat([g, history_out], dim=-1))

    def clone(self):
        cloned_net = ReptileModel_Encoder(self.in_size,
                                  self.n_targets,
                                  self.history_in_size,
                                  self.history_out_size,
                                  self.encoder_layer_sizes,
                                  self.encoder_nonlin,
                                  self.layer_sizes,
                                  self.nonlin_string)
        cloned_net.load_state_dict(self.state_dict())
        if self.is_cuda():
            cloned_net.cuda()
        return cloned_net


# Now that network accepts a single cat'ed batch, don't need
# a separate method for "history"
def reptile_inner_opt(network,
                      inner_optimizer,
                      inner_loss,
                      input_batch,
                      target_batch,
                      K,
                      num_steps,
                      random=True):
    for step in range(num_steps):
        inner_optimizer.zero_grad()  # should I do this?
        if random:
            train_idxs = np.random.choice(input_batch.size(0), K, replace=False)
        else:
            train_idxs = np.arange(K)
        train_input_batch = input_batch[train_idxs]
        train_target_batch = target_batch[train_idxs]
        net_out = network(train_input_batch)
        l = inner_loss(net_out, train_target_batch)
        l.backward()
        inner_optimizer.step()
    full_batch_output = network(input_batch)
    full_batch_final_loss = inner_loss(full_batch_output, target_batch)
    unseen_batch_output = network(torch_delete_batch_idxs(input_batch, train_idxs))
    unseen_batch_final_loss = inner_loss(unseen_batch_output, torch_delete_batch_idxs(target_batch, train_idxs))
    return network, full_batch_final_loss.item(), unseen_batch_final_loss.item()  # return final loss

# def reptile_inner_opt_history(network,
#                               inner_optimizer,
#                               inner_loss,
#                               input_batch,
#                               history_batch,
#                               target_batch,
#                               K,
#                               num_steps,
#                               random=True):
#     for step in range(num_steps):
#         inner_optimizer.zero_grad()  # should I do this?
#         if random:
#             train_idxs = np.random.choice(input_batch.size(0), K, replace=False)
#         else:
#             train_idxs = np.arange(K)
#         train_input_batch = input_batch[train_idxs]
#         train_history_batch = history_batch[train_idxs]
#         train_target_batch = target_batch[train_idxs]
#         net_out = network(train_input_batch, train_history_batch)
#         l = inner_loss(net_out, train_target_batch)
#         l.backward()
#         inner_optimizer.step()
#     full_batch_output = network(input_batch, history_batch)
#     full_batch_final_loss = inner_loss(full_batch_output, target_batch)
#     unseen_batch_output = network(torch_delete_batch_idxs(input_batch, train_idxs),
#                                  torch_delete_batch_idxs(history_batch, train_idxs))
#     unseen_batch_final_loss = inner_loss(unseen_batch_output, torch_delete_batch_idxs(target_batch, train_idxs))
#     return network, full_batch_final_loss.item(), unseen_batch_final_loss.item()  # return final loss

# No noisy for now.
def train_reptile_net(train_dataloader,
                      meta_net,
                      meta_optimizer,
                      inner_lr,
                      num_inner_steps,
                      input_batch_idxs,
                      target_batch_idx_gt,
                      random_K_training,
                      K,
                      device):

    avg_train_loss = 0
    for i, task_batch in enumerate(train_dataloader):
        inner_net = meta_net.clone()
        optimizer = torch.optim.Adam(inner_net.parameters(), lr=inner_lr)
        inner_loss = torch.nn.MSELoss()
        # inner_loss = torch.nn.L1Loss()
        batches = task_batch
        input_batches = []
        for idx in input_batch_idxs:
            input_batches.append(batches[idx])
        input_batch = torch.cat(input_batches, dim=-1)
        gt_batch = batches[target_batch_idx_gt]
        if random_K_training:
            num = np.random.randint(1, input_batch.size(0)-1)
        else:
            num = K
        inner_net, final_loss, unseen_loss = reptile_inner_opt(inner_net,
                                                               optimizer,
                                                               inner_loss,
                                                               input_batch.float().to(device),
                                                               gt_batch.float().to(device),
                                                               K = num,
                                                               num_steps=num_inner_steps)

        avg_train_loss += final_loss
        meta_net.point_grad_to(inner_net)
        meta_optimizer.step()
    return avg_train_loss/len(train_dataloader)


def val_reptile_net(val_dataloader,
                    meta_net,
                    inner_lr,
                    num_inner_steps,
                    input_batch_idxs,
                    target_batch_idx_gt,
                    K,
                    device):
    avg_val_task_loss = 0
    avg_val_unseen_loss = 0
    for i, task_batch in enumerate(val_dataloader):
        inner_net = meta_net.clone()
        optimizer = torch.optim.Adam(inner_net.parameters(), lr=inner_lr)
        inner_loss = torch.nn.MSELoss()
        # inner_loss = torch.nn.L1Loss()
        batches = task_batch
        input_batches = []
        for idx in input_batch_idxs:
            input_batches.append(batches[idx])
        input_batch = torch.cat(input_batches, dim=-1)
        gt_batch = batches[target_batch_idx_gt]
        inner_net, final_loss, unseen_loss = reptile_inner_opt(inner_net,
                                                               optimizer,
                                                               inner_loss,
                                                               input_batch.float().to(device),
                                                               gt_batch.float().to(device),
                                                               K = K,
                                                               num_steps=num_inner_steps,
                                                               random=False)
        avg_val_task_loss += final_loss
        avg_val_unseen_loss += unseen_loss
    avg_val_task_loss /= len(val_dataloader)
    avg_val_unseen_loss /= len(val_dataloader)
    return avg_val_task_loss, avg_val_unseen_loss


def adapt_reptile(in_data, target_data, meta_net,
                  inner_lr, num_inner_steps):
    device = next(meta_net.parameters()).device
    inner_net = meta_net.clone()
    optimizer = torch.optim.Adam(inner_net.parameters(), lr=inner_lr)
    inner_loss = torch.nn.MSELoss()
    inner_net, final_loss, unseen_loss = reptile_inner_opt(inner_net,
                                                           optimizer,
                                                           inner_loss,
                                                           in_data.float().to(device),
                                                           target_data.float().to(device),
                                                           K=in_data.size(0),
                                                           num_steps=num_inner_steps)
    # optimizer.zero_grad()
    return inner_net
