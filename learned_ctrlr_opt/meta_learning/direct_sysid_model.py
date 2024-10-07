import torch
from torch import nn
from learned_ctrlr_opt.utils.learning_utils import create_network

# model that maps from state, action history to sysid params
# If doing direct sysid, then how do you handle MinMax scaling?
# would maybe have to do StandardScaling, then...
# I would also expect direct sysid to perform the worst at OOB generalization
class DirectSysIdModel(nn.Module):
    def __init__(self, history_dim,
                 sysid_dim, layer_sizes, nonlin="relu"):
        super().__init__()
        self.model = create_network(history_dim,
                                    sysid_dim,
                                    layer_sizes,
                                    nonlin)

    def forward(self, x):
        return self.model(x)


def train_direct_sysid_model(train_dataloader,
                             network: DirectSysIdModel,
                             optimizer,
                             criterion,
                             history_batch_idx,
                             target_batch_idx,
                             device):
    avg_train_residual = 0
    for i, batches in enumerate(train_dataloader):
        optimizer.zero_grad()
        net_in = batches[history_batch_idx].float().to(device)
        target = batches[target_batch_idx].to(device)
        net_out = network(net_in)
        loss = criterion(net_out, target)
        loss.backward()
        optimizer.step()
        avg_train_residual += loss
    return avg_train_residual / len(train_dataloader)

def val_direct_sysid_model(val_dataloader,
                        network,
                        loss,
                        history_batch_idx,
                        target_batch_idx,
                        device):
    avg_val_loss = 0
    for i, task_batch in enumerate(val_dataloader):
        batches = task_batch
        net_in = batches[history_batch_idx].float().to(device)
        gt_batch = batches[target_batch_idx].to(device)
        with torch.no_grad():
            net_out = network(net_in)
        avg_val_loss += loss(net_out, gt_batch)
    return avg_val_loss / len(val_dataloader)
