import torch.optim
import os
from omegaconf import OmegaConf

from learned_ctrlr_opt.meta_learning.lsr_net import *
from learned_ctrlr_opt.meta_learning.reptile_net import *
from learned_ctrlr_opt.meta_learning.teacher_student import *
from learned_ctrlr_opt.meta_learning.utils import get_scalers


def load_ts_and_scalers(experiment_cfg, abs_path_header=None):
    if abs_path_header is None:
        ts_checkpoint_dir = experiment_cfg.ts_ckpt_dir
    else:
        ts_checkpoint_dir = os.path.join(abs_path_header, experiment_cfg.ts_ckpt_dir)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ts_cfg = OmegaConf.load(os.path.join(ts_checkpoint_dir, "config.yaml"))
    # load scalers
    if abs_path_header is None:
        scalers = get_scalers(ts_cfg.path_to_dataset, ts_cfg.history_length, ts_cfg.metric_idxs,
                              ts_cfg.metric_idxs_to_invert)
    else:
        scalers = get_scalers(os.path.join(abs_path_header, ts_cfg.path_to_dataset),
                              ts_cfg.history_length,
                              ts_cfg.metric_idxs,
                              ts_cfg.metric_idxs_to_invert)
    history_scaler = scalers[-1]
    gain_scaler = scalers[0]
    metric_scaler = scalers[2]

    gain_dim = len(ts_cfg.gains_to_optimize)
    lookahead_dim = experiment_cfg.ts_lookahead_dim
    history_in_size = ts_cfg.history_length * ts_cfg.traj_dim
    theta_dim = len(ts_cfg.thetas_randomized)

    network = TeacherStudentSysIDModel(gain_dim,
                                       lookahead_dim,
                                       theta_dim+ts_cfg.traj_dim,
                                       history_in_size,
                                       ts_cfg.latent_dim,
                                       len(ts_cfg.metric_idxs),
                                       ts_cfg.layer_sizes,
                                       ts_cfg.teacher_layers,
                                       ts_cfg.student_layers,
                                       ts_cfg.nonlin).to(device)
    network.load_state_dict(torch.load(os.path.join(ts_checkpoint_dir, experiment_cfg.ts_ckpt_file), map_location=device))
    network.use_teacher = False
    return network, gain_scaler, history_scaler, metric_scaler


def load_kf_and_scalers(experiment_cfg, abs_path_header=None):
    if abs_path_header is None:
        kf_checkpoint_dir = experiment_cfg.kf_ckpt_dir
    else:
        kf_checkpoint_dir = os.path.join(abs_path_header, experiment_cfg.kf_ckpt_dir)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    kf_cfg = OmegaConf.load(os.path.join(kf_checkpoint_dir, "config.yaml"))
    # load scalers
    if abs_path_header is None:
        scalers = get_scalers(kf_cfg.path_to_dataset, kf_cfg.history_length, kf_cfg.metric_idxs,
                              kf_cfg.metric_idxs_to_invert)
    else:
        scalers = get_scalers(os.path.join(abs_path_header, kf_cfg.path_to_dataset),
                              kf_cfg.history_length,
                              kf_cfg.metric_idxs,
                              kf_cfg.metric_idxs_to_invert)
    history_scaler = scalers[-1]
    gain_scaler = scalers[0]
    metric_scaler = scalers[2]

    gain_dim = len(kf_cfg.gains_to_optimize)
    lookahead_dim = experiment_cfg.lookahead_dim
    history_in_size = kf_cfg.history_length * kf_cfg.traj_dim
    if kf_cfg.history_length > 0:
        kf_network = LSRBasisNet_encoder(gain_in_size=gain_dim + lookahead_dim,
                                         n_bases=kf_cfg.n_bases,
                                         n_targets=len(kf_cfg.metric_idxs),
                                         history_in_size=history_in_size,
                                         history_out_size=kf_cfg.history_latent_dim,
                                         encoder_layer_sizes=kf_cfg.encoder_layer_sizes,
                                         encoder_nonlin=kf_cfg.encoder_nonlin,
                                         layer_sizes=kf_cfg.layer_sizes,
                                         nonlin=kf_cfg.nonlin).float().train().to(device)
    else:
        kf_network = LSRBasisNet(in_size=gain_dim + lookahead_dim,
                                 n_bases=kf_cfg.n_bases,
                                 n_targets=len(kf_cfg.metric_idxs),
                                 layer_sizes=kf_cfg.layer_sizes,
                                 nonlin=kf_cfg.nonlin).float().train().to(device)

    kf_network.load_state_dict(torch.load(os.path.join(kf_checkpoint_dir, experiment_cfg.kf_ckpt_file), map_location=device))
    kf_network.use_last_layer = False
    kf_network = kf_network.eval().to(device)
    return kf_network, gain_scaler, history_scaler, metric_scaler

def load_reptile_and_scalers(experiment_cfg):
    reptile_checkpoint_dir = experiment_cfg.reptile_ckpt_dir
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    reptile_cfg = OmegaConf.load(os.path.join(reptile_checkpoint_dir, "config.yaml"))
    scalers = get_scalers(reptile_cfg.path_to_dataset, reptile_cfg.history_length, reptile_cfg.metric_idxs,
                          reptile_cfg.metric_idxs_to_invert)
    history_scaler = scalers[-1]
    gain_scaler = scalers[0]
    metric_scaler = scalers[2]

    gain_dim = len(reptile_cfg.gains_to_optimize)
    lookahead_dim = experiment_cfg.lookahead_dim
    history_in_size = reptile_cfg.history_length * reptile_cfg.traj_dim

    if reptile_cfg.history_length > 0:
        reptile_net = ReptileModel_Encoder(
            in_size=gain_dim + lookahead_dim,
            n_targets=len(reptile_cfg.metric_idxs),
            history_in_size=history_in_size,
            history_out_size=reptile_cfg.history_latent_dim,
            encoder_layer_sizes=reptile_cfg.encoder_layer_sizes,
            encoder_nonlin=reptile_cfg.encoder_nonlin,
            layer_sizes=reptile_cfg.layer_sizes,
            nonlin=reptile_cfg.nonlin).float().to(device)
    else:
        reptile_net = ReptileModel(
            in_size = gain_dim + lookahead_dim,
            n_targets=len(reptile_cfg.metric_idxs),
            layer_sizes=reptile_cfg.layer_sizes,
            nonlin=reptile_cfg.nonlin).float().to(device)

    reptile_net.load_state_dict(torch.load(os.path.join(reptile_checkpoint_dir, "model_epoch_25.pt")))
    return reptile_net, gain_scaler, history_scaler, metric_scaler

