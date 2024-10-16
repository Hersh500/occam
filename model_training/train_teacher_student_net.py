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

from learned_ctrlr_opt.meta_learning.teacher_student import *
from learned_ctrlr_opt.meta_learning.utils import get_sysid_history_dataset


@hydra.main(version_base=None, config_path="configs/meta_learning_confs", config_name="ts_conf.yaml")
def main(cfg):
    train_dataset, train_dataloader, val_dataset, val_dataloader = get_sysid_history_dataset(
        cfg,
        param_types[cfg.params_type].get_bounds(),
        batched=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    seeds = cfg.seeds
    if isinstance(seeds, int):
        seeds = [seeds]

    total_size = 0
    test_batch = train_dataset[0]
    for i in cfg.student_input_batch_idxs:
        total_size += test_batch[i].shape[-1]
        print(test_batch[i].shape[-1])
    print(f"Total input size of network is {total_size}")
    for seed in seeds:
        history_in_size = cfg.history_length * cfg.traj_dim
        gain_and_traj_in_size = total_size - history_in_size
        gain_dim = len(cfg.gains_to_optimize)
        task_dim = gain_and_traj_in_size - gain_dim
        theta_dim = len(cfg.thetas_randomized)
        network = TeacherStudentSysIDModel(gain_dim,
                                           task_dim,
                                           theta_dim,
                                           history_in_size,
                                           cfg.latent_dim,
                                           len(cfg.metric_idxs),
                                           cfg.layer_sizes,
                                           cfg.teacher_layers,
                                           cfg.student_layers,
                                           cfg.nonlin,
                                           cfg.direct).to(device)
        np.random.seed(seed)
        network.apply(init_weights)
        if cfg.pretrained_model_ckpt is not None:
            print(f"Loading checkpoint from {cfg.pretrained_model_ckpt}")
            network.load_state_dict(torch.load(cfg.pretrained_model_ckpt))
        network.use_teacher = True

        teacher_opt = torch.optim.Adam(network.parameters(), lr=cfg.phase_1_lr)
        wandb.init(project="control-adaptation-lstsq",
                   config=OmegaConf.to_container(cfg, resolve=True))

        date_string = datetime.now().strftime("%b_%d_%Y")

        checkpoint_dir = os.path.join("model_checkpoints", cfg.robot + "_ts_net", wandb.run.name + "_" + date_string)
        print(f"Saving checkpoints to {checkpoint_dir}")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        OmegaConf.save(cfg, f=os.path.join(checkpoint_dir, "config.yaml"))
        loss = torch.nn.MSELoss()

        try:
            for i in range(cfg.num_phase_1_iters):
                if i % 5 == 0:
                    avg_val_residual = val_teacher(val_dataloader,
                                                   network,
                                                   loss,
                                                   cfg.teacher_input_batch_idxs,
                                                   cfg.history_batch_idx,
                                                   cfg.traj_dim,
                                                   cfg.target_batch_idx,
                                                   device)
                    print(f"on epoch {i}, avg val residual was {avg_val_residual}")
                    torch.save(network.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{i}.pt"))
                avg_train_residual = train_teacher(train_dataloader,
                                                   network,
                                                   teacher_opt,
                                                   loss,
                                                   cfg.teacher_input_batch_idxs,
                                                   cfg.history_batch_idx,
                                                   cfg.traj_dim,
                                                   cfg.target_batch_idx,
                                                   device)

                wandb.log({"avg_train_residual": avg_train_residual,
                           "avg_val_residual": avg_val_residual})

                print(f"epoch {i}: train residual = {avg_train_residual}")

            print("switching to student training")
            network.use_teacher = False
            student_optimizer = torch.optim.Adam(network.student_encoder.parameters(), lr=cfg.phase_2_lr)

            for i in range(cfg.num_phase_2_iters):
                if i % 5 == 0:
                    avg_val_residual = val_student(val_dataloader,
                                                   network,
                                                   loss,
                                                   cfg.student_input_batch_idxs,
                                                   cfg.target_batch_idx,
                                                   device)

                    print(f"on epoch {i}, avg val residual was {avg_val_residual}")
                    torch.save(network.state_dict(),
                               os.path.join(checkpoint_dir, f"model_epoch_{i + cfg.num_phase_1_iters}.pt"))

                avg_student_residual = train_student(train_dataloader,
                                                     network,
                                                     student_optimizer,
                                                     loss,
                                                     cfg.theta_batch_idx,
                                                     cfg.history_batch_idx,
                                                     cfg.traj_dim,
                                                     device)

                wandb.log({"avg_student_residual": avg_student_residual,
                           "avg_val_residual": avg_val_residual})
                print(f"epoch {i}: student residual = {avg_student_residual}")
        finally:
            wandb.finish()


if __name__ == "__main__":
    main()
