import numpy as np
import torch
from torch.nn import Linear
from torch import nn

def init_weights(m):
    if isinstance(m, Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def torch_delete_batch_idxs(arr, idxs):
    idxs_except = [i for i in range(arr.size(0)) if i not in idxs]
    return arr[idxs_except, ...]

def create_network(in_size, out_size, layer_sizes, nonlin_str):
    if nonlin_str.lower() == "relu":
        nonlin = nn.ReLU
    elif nonlin_str.lower() == "sigmoid":
        nonlin = nn.Sigmoid
    else:
        raise NotImplementedError()
    net = nn.Sequential(nn.Linear(in_size, layer_sizes[0]), nonlin())
    for i, s in enumerate(layer_sizes[:-1]):
        net.append(nn.Linear(s, layer_sizes[i + 1]))
        net.append(nonlin())
    net.append(nn.Linear(layer_sizes[-1], out_size))
    return net