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

from learned_ctrlr_opt.meta_learning.lsr_net import *
from learned_ctrlr_opt.meta_learning.reptile_net import *
from learned_ctrlr_opt.meta_learning.basis_kf import train_basis_net_kf, eval_basis_net_kf
from learned_ctrlr_opt.meta_learning.utils import get_sysid_history_dataset


@hydra.main(version_base=None, config_path="configs/meta_learning_confs", config_name="lsr_conf.yaml")
def main(cfg):
    train_dataset, train_dataloader, val_dataset, val_dataloader = get_sysid_history_dataset(
        cfg,
        param_types[cfg.params_type].get_bounds(),
        True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    seeds = cfg.seeds
    if isinstance(seeds, int):
        seeds = [seeds]

    total_size = 0
    test_batch = train_dataset[0]
    for i in cfg.input_batch_idxs:
        total_size += test_batch[i].shape[-1]
        print(test_batch[i].shape[-1])
    print(f"Total input size of network is {total_size}")
    for seed in seeds:
        if cfg.history_length > 0:
            history_in_size = cfg.history_length * cfg.traj_dim
            gain_and_traj_in_size = total_size - history_in_size
            print(gain_and_traj_in_size)
            network = LSRBasisNet_encoder(gain_in_size=gain_and_traj_in_size,
                                          n_bases=cfg.n_bases,
                                          n_targets=len(cfg.metric_idxs),
                                          history_in_size=history_in_size,
                                          history_out_size=cfg.history_latent_dim,
                                          encoder_layer_sizes=cfg.encoder_layer_sizes,
                                          encoder_nonlin=cfg.encoder_nonlin,
                                          layer_sizes=cfg.layer_sizes,
                                          nonlin=cfg.nonlin).float().train().to(device)
        else:
            network = LSRBasisNet(in_size=total_size,
                                  n_bases=cfg.n_bases,
                                  n_targets=len(cfg.metric_idxs),
                                  layer_sizes=cfg.layer_sizes,
                                  nonlin=cfg.nonlin).float().train().to(device)
        network.use_last_layer = True
        np.random.seed(seed)
        network.apply(init_weights)
        if cfg.pretrained_model_ckpt is not None:
            print(f"Loading checkpoint from {cfg.pretrained_model_ckpt}")
            network.load_state_dict(torch.load(cfg.pretrained_model_ckpt))

        optimizer1 = torch.optim.Adam(network.parameters(), lr=cfg.phase_1_lr)
        wandb.init(project="control-adaptation-lstsq",
                   config=OmegaConf.to_container(cfg, resolve=True))

        date_string = datetime.now().strftime("%b_%d_%Y")

        checkpoint_dir = os.path.join("model_checkpoints", cfg.robot + "_lstsq_net", wandb.run.name + "_" + date_string)
        print(f"Saving checkpoints to {checkpoint_dir}")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        OmegaConf.save(cfg, f=os.path.join(checkpoint_dir, "config.yaml"))

        try:
            for i in range(cfg.num_phase_1_iters):
                if i % 5 == 0:
                    if cfg.phase_1_method.lower() == "linear":
                        avg_val_residual = eval_basis_net_no_adapt(val_dataloader,
                                                                   network,
                                                                   cfg.input_batch_idxs,
                                                                   cfg.target_batch_idx,
                                                                   device)
                        avg_val_residual_unseen = None
                    elif cfg.phase_1_method.lower() == "reptile":
                        avg_val_residual, avg_val_residual_unseen = eval_basis_net_reptile(val_dataloader,
                                                                                           network,
                                                                                           cfg.inner_lr,
                                                                                           cfg.num_inner_steps,
                                                                                           cfg.input_batch_idxs,
                                                                                           cfg.target_batch_idx,
                                                                                           device,
                                                                                           cfg.K)
                    else:
                        raise ValueError(f"{cfg.phase_1_method} is not a valid method")
                    print(f"on epoch {i}, avg val residual was {avg_val_residual}")
                    print(f"on epoch {i}, avg val residual unseen was {avg_val_residual_unseen}")
                    torch.save(network.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{i}.pt"))
                if cfg.phase_1_method.lower() == "linear":
                    avg_train_residual = pretrain_basis_net_no_adapt(train_dataloader,
                                                                     network,
                                                                     optimizer1,
                                                                     cfg.input_batch_idxs,
                                                                     cfg.target_batch_idx, device)
                    wandb.log({"avg_train_residual": avg_train_residual,
                               "avg_val_residual": avg_val_residual})
                elif cfg.phase_1_method.lower() == "reptile":
                    avg_train_residual = pretrain_basis_net_reptile(train_dataloader,
                                                                    network,
                                                                    optimizer1,
                                                                    cfg.inner_lr,
                                                                    cfg.num_inner_steps,
                                                                    cfg.input_batch_idxs,
                                                                    cfg.target_batch_idx,
                                                                    cfg.random_num_lstsq,
                                                                    device)
                    wandb.log({"avg_train_residual": avg_train_residual,
                               "avg_val_residual": avg_val_residual,
                               "avg_val_residual_unseen": avg_val_residual_unseen})
                else:
                    raise ValueError(f"{cfg.phase_1_method} is not a valid method")

                print(f"epoch {i}: train residual = {avg_train_residual}")

            network.use_last_layer = False
            print("switching to phase 2")
            if cfg.pretrained_model_ckpt is None or cfg.num_phase_1_iters > 0:
                network.initialize_priors()
            network = network.to(device)
            optimizer2 = torch.optim.Adam(network.parameters(), lr=cfg.phase_2_lr)

            for i in range(cfg.num_phase_2_iters):
                if i % 5 == 0:
                    if cfg.phase_2_method == "lstsq":
                        avg_val_residual, avg_val_residual_unseen = eval_basis_net_lstsq(val_dataloader,
                                                                                         network,
                                                                                         cfg.input_batch_idxs,
                                                                                         cfg.target_batch_idx,
                                                                                         device)
                    else:
                        avg_val_residual, avg_val_residual_unseen = eval_basis_net_kf(val_dataloader,
                                                                                      network,
                                                                                      cfg.input_batch_idxs,
                                                                                      cfg.target_batch_idx,
                                                                                      device)

                    print(f"on epoch {i}, avg val residual was {avg_val_residual}")
                    print(f"on epoch {i}, avg unseen val residual was {avg_val_residual_unseen}")
                    torch.save(network.state_dict(),
                               os.path.join(checkpoint_dir, f"model_epoch_{i + cfg.num_phase_1_iters}.pt"))

                if cfg.phase_2_method == "lstsq":
                    avg_train_residual = train_basis_net_lstsq(train_dataloader,
                                                               network,
                                                               optimizer2,
                                                               cfg.input_batch_idxs,
                                                               cfg.target_batch_idx,
                                                               cfg.random_num_lstsq, device)
                else:
                    avg_train_residual = train_basis_net_kf(train_dataloader,
                                                            network,
                                                            optimizer2,
                                                            cfg.input_batch_idxs,
                                                            cfg.target_batch_idx,
                                                            cfg.random_num_lstsq, device)

                wandb.log({"avg_train_residual": avg_train_residual,
                           "avg_val_residual": avg_val_residual,
                           "avg_val_residual_unseen": avg_val_residual_unseen})
                print(f"epoch {i}: train residual = {avg_train_residual}")
        finally:
            wandb.finish()


if __name__ == "__main__":
    main()
