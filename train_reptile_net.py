from learned_ctrlr_opt.systems.utils import param_types
import numpy as np
import h5py
import torch.optim
from datetime import datetime
import wandb
import os
import hydra
from omegaconf import OmegaConf

from learned_ctrlr_opt.meta_learning.reptile_net import *
from learned_ctrlr_opt.meta_learning.utils import get_sysid_history_dataset
from learned_ctrlr_opt.utils.learning_utils import init_weights, torch_delete_batch_idxs, create_network

@hydra.main(version_base=None, config_path="configs/meta_learning_confs", config_name="reptile_conf.yaml")
def main(cfg):
    train_dataset, train_dataloader, val_dataset, val_dataloader = get_sysid_history_dataset(
        cfg,
        param_types[cfg.params_type].get_bounds(),
        True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    gain_dim = train_dataset[0][0].shape[-1]

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
            gain_and_traj_size = total_size - history_in_size
            meta_net = ReptileModel_Encoder(
                        in_size = gain_and_traj_size,
                        n_targets=len(cfg.metric_idxs),
                        history_in_size=history_in_size,
                        history_out_size=cfg.history_latent_dim,
                        encoder_layer_sizes=cfg.encoder_layer_sizes,
                        encoder_nonlin=cfg.encoder_nonlin,
                        layer_sizes=cfg.layer_sizes,
                        nonlin=cfg.nonlin).float().to(device)
        else:
            meta_net = ReptileModel(
                in_size = total_size,
                n_targets=len(cfg.metric_idxs),
                layer_sizes=cfg.layer_sizes,
                nonlin=cfg.nonlin
            ).float().to(device)
            
        np.random.seed(seed)
        meta_net.apply(init_weights)
        meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=cfg.meta_lr)
        wandb.init(project="control-adaptation-lstsq",
                   config=OmegaConf.to_container(cfg, resolve=True))
        date_string = datetime.now().strftime("%b_%d_%Y")

        checkpoint_dir = os.path.join("model_checkpoints", cfg.robot + "_reptile_net", wandb.run.name + "_" + date_string)
        print(f"Saving checkpoints to {checkpoint_dir}")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        OmegaConf.save(cfg, f=os.path.join(checkpoint_dir, "config.yaml"))

        try:
            for i in range(cfg.num_meta_iters):
                if i % 5 == 0:
                    avg_val_residual, avg_val_residual_unseen = val_reptile_net(val_dataloader, 
                                                                                meta_net,
                                                                                cfg.inner_lr,
                                                                                cfg.num_inner_steps,
                                                                                cfg.input_batch_idxs,
                                                                                cfg.target_batch_idx,
                                                                                cfg.K,
                                                                                device)
                    print(f"on epoch {i}, avg val residual was {avg_val_residual}")
                    print(f"on epoch {i}, avg unseen val residual was {avg_val_residual_unseen}")
                    torch.save(meta_net.state_dict(),
                               os.path.join(checkpoint_dir, f"model_epoch_{i}.pt"))
                avg_train_residual = train_reptile_net(train_dataloader,
                                                       meta_net,
                                                       meta_optimizer,
                                                       cfg.inner_lr,
                                                       cfg.num_inner_steps,
                                                       cfg.input_batch_idxs,
                                                       cfg.target_batch_idx,
                                                       cfg.random_K_training,
                                                       cfg.K,
                                                       device)
                wandb.log({"avg_train_residual": avg_train_residual,
                           "avg_val_residual": avg_val_residual,
                           "avg_val_residual_unseen": avg_val_residual_unseen})
                print(f"epoch {i}: train residual = {avg_train_residual}")
        finally:
            wandb.finish()


if __name__ == "__main__":
    main()
