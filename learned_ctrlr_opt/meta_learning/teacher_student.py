import torch
from torch import nn
from learned_ctrlr_opt.utils.learning_utils import create_network
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym.spaces as spaces

class TeacherStudentSysIDModel(nn.Module):
    def __init__(self,
                 gain_dim,
                 task_dim,
                 sysid_dim,
                 history_dim,
                 latent_dim,
                 n_targets,
                 layer_sizes,
                 gt_encoder_layers,
                 student_encoder_layers,
                 nonlin="relu",
                 direct=False):
        super().__init__()
        self.use_teacher = True
        if not direct:
            self.teacher_encoder = create_network(sysid_dim,
                                                  latent_dim,
                                                  gt_encoder_layers,
                                                  nonlin)
            self.latent_dim = latent_dim
        else:
            print("Direct SYSID! teacher encoder is simply passthrough")
            self.teacher_encoder = nn.Identity()
            self.latent_dim = sysid_dim
        self.student_encoder = create_network(history_dim,
                                              self.latent_dim,
                                              student_encoder_layers,
                                              nonlin)
        self.model = create_network(gain_dim+task_dim+self.latent_dim,
                                    n_targets,
                                    layer_sizes,
                                    nonlin)
        self.gain_dim = gain_dim
        self.task_dim = task_dim
        self.sysid_dim = sysid_dim

    def forward(self, x):
        gain_and_task = x[...,:self.gain_dim+self.task_dim]
        context_info = x[..., self.gain_dim + self.task_dim:]
        if self.use_teacher:
            latent = self.teacher_encoder(context_info)
        else:
            latent = self.student_encoder(context_info)
        # concatenate
        concat = torch.cat([gain_and_task, latent], dim=-1)
        return self.model(concat)

    def encode_student(self, history):
        return self.student_encoder(history)

    def encode_teacher(self, gt_theta):
        return self.teacher_encoder(gt_theta)


def val_student(val_dataloader,
                network,
                loss,
                input_batch_idxs,
                target_batch_idx_gt,
                device):
    avg_val_loss = 0
    for i, task_batch in enumerate(val_dataloader):
        batches = task_batch
        input_batches = []
        for idx in input_batch_idxs:
            input_batches.append(batches[idx])
        input_batch = torch.cat(input_batches, dim=-1).to(device)
        gt_batch = batches[target_batch_idx_gt].to(device)
        with torch.no_grad():
            net_out = network(input_batch)
        avg_val_loss += loss(net_out, gt_batch)
    return avg_val_loss / len(val_dataloader)


def val_teacher(val_dataloader,
                network,
                loss,
                input_batch_idxs,
                history_batch_idx,
                traj_dim,
                target_batch_idx_gt,
                device):
    avg_val_loss = 0
    for i, task_batch in enumerate(val_dataloader):
        batches = task_batch
        input_batches = []
        for idx in input_batch_idxs:
            if idx == history_batch_idx:
                input_batches.append(batches[idx][...,-traj_dim:])
            else:
                input_batches.append(batches[idx])
        input_batch = torch.cat(input_batches, dim=-1).float().to(device)
        gt_batch = batches[target_batch_idx_gt].to(device)
        with torch.no_grad():
            net_out = network(input_batch)
        avg_val_loss += loss(net_out, gt_batch)
    return avg_val_loss / len(val_dataloader)


def train_teacher(train_dataloader,
                  network: TeacherStudentSysIDModel,
                  optimizer,
                  criterion,
                  input_batch_idxs,
                  history_batch_idx,
                  traj_dim,
                  target_batch_idx_gt,
                  device):
    network.use_teacher = True
    avg_train_residual = 0
    for i, batches in enumerate(train_dataloader):
        optimizer.zero_grad()
        input_batches = []
        for idx in input_batch_idxs:
            if idx == history_batch_idx:
                input_batches.append(batches[idx][...,-traj_dim:])
            else:
                input_batches.append(batches[idx])
        net_in = torch.cat(input_batches, dim=-1).float().to(device)
        target = batches[target_batch_idx_gt].to(device)
        net_out = network(net_in)
        loss = criterion(net_out, target)
        loss.backward()
        optimizer.step()
        avg_train_residual += loss
    return avg_train_residual / len(train_dataloader)


def train_student(theta_dataloader,
                  network: TeacherStudentSysIDModel,
                  student_optimizer,
                  criterion,
                  theta_batch_idx,
                  history_batch_idx,
                  traj_dim,
                  device):
    avg_train_residual = 0
    for i, batches in enumerate(theta_dataloader):
        student_optimizer.zero_grad()
        theta_batch = batches[theta_batch_idx]
        # making some hacky assumptions here...
        # teacher_input = torch.cat([batches[history_batch_idx][...,-traj_dim:], theta_batch], dim=-1).float().to(device)
        teacher_input = theta_batch.float().to(device)
        with torch.no_grad():
            teacher_output = network.encode_teacher(teacher_input)
        student_output = network.encode_student(batches[history_batch_idx].float().to(device))
        loss = criterion(student_output, teacher_output)
        loss.backward()
        student_optimizer.step()
        avg_train_residual += loss
    return avg_train_residual / len(theta_dataloader)


class TeacherRLFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: spaces.Box,
                 task_dim,
                 traj_dim,
                 theta_dim,
                 sysid_latent_dim,
                 network_architecture,
                 teacher_net_arch,
                 features_dim):
        super().__init__(observation_space, features_dim)
        self.task_dim = task_dim
        self.traj_dim = traj_dim
        self.theta_dim = theta_dim
        self.sysid_latent_dim = sysid_latent_dim
        self.network_architecture = network_architecture
        self.theta_network = create_network(theta_dim, sysid_latent_dim, teacher_net_arch, "relu")
        self.extractor_network = create_network(traj_dim+task_dim+sysid_latent_dim, features_dim,
                                                network_architecture, "relu")

    def forward(self, observations):
        theta = observations[...,-self.theta_dim:]
        theta_latent = self.theta_network(theta)
        other_input = torch.cat([observations[...,:-self.theta_dim], theta_latent], dim=-1)
        return self.extractor_network(other_input)

    def encode_theta(self, theta):
        return self.theta_network(theta)
