from learned_ctrlr_opt.systems.utils import param_types  # has to be first due to isaac gym requirements
import numpy as np
import h5py
import torch.optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader
from learned_ctrlr_opt.utils.learning_utils import init_weights
from datetime import datetime
import wandb
import os
import hydra
from omegaconf import OmegaConf
import sys

from meta_bo.models import FPACOH_MAP_GP
from meta_bo.domain import ContinuousDomain
from learned_ctrlr_opt.meta_learning.utils import get_sysid_history_dataset


@hydra.main(version_base=None, config_path="configs/meta_learning_confs", config_name="fpacoh_conf.yaml")
def main(cfg):
    train_dataset, train_dataloader, val_dataset, val_dataloader = get_sysid_history_dataset(
        cfg,
        param_types[cfg.params_type].get_bounds(),
        True)
    meta_train_tuples = []
    for i in range(len(train_dataset)):
        batches = train_dataset[i]
        input_batch = []
        output_batch = batches[cfg.target_batch_idx].numpy()
        for idx in cfg.input_batch_idxs:
            input_batch.append(batches[idx])
        input_batch = torch.cat(input_batch, dim=-1)
        meta_train_tuples.append((input_batch.numpy(), output_batch))
    input_dim = len(cfg.gains_to_optimize)
    domain = ContinuousDomain(np.zeros(input_dim), np.ones(input_dim))
    fpacoh_model = FPACOH_MAP_GP(domain=domain,
                                 num_iter_fit=cfg.num_iters,
                                 weight_decay=1e-4,
                                 prior_factor=cfg.prior_factor,
                                 task_batch_size=meta_train_tuples[0][0].shape[0],
                                 feature_dim=cfg.feature_dim,
                                 mean_nn_layers=cfg.mean_nn_layer_size,
                                 kernel_nn_layers=cfg.kernel_nn_layer_size)
    fpacoh_model.meta_fit(meta_train_tuples, log_period=20)
    date_string = datetime.now().strftime("%b_%d_%Y")
    checkpoint_dir = os.path.join("model_checkpoints", cfg.robot + "_fpacoh", date_string)
    torch.save(fpacoh_model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
    OmegaConf.save(cfg, f=os.path.join(checkpoint_dir, "config.yaml"))


if __name__ == "__main__":
    main()
