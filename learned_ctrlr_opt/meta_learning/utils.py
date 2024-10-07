import numpy as np
import h5py
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from learned_ctrlr_opt.utils.dataset_utils import H5IntrinsicBatchedDataset, get_idxs_in_bounds, \
    get_idxs_out_of_bounds, H5NonBatchedDataset


def get_scalers(path_to_dataset, history_length, metric_idxs, metric_idxs_to_invert=[]):
    dset_f = h5py.File(path_to_dataset, 'r')
    all_gains = np.array(dset_f["gains"])
    gain_dim = all_gains.shape[-1]
    all_metrics = np.array(dset_f["metrics"])
    all_metrics[..., metric_idxs_to_invert] = 1/(1 + all_metrics[..., metric_idxs_to_invert])
    all_thetas = np.array(dset_f["intrinsics"])
    theta_dim = all_thetas.shape[-1]

    gain_scaler = MinMaxScaler().fit(all_gains.reshape(-1, all_gains.shape[-1]))
    theta_scaler = MinMaxScaler().fit(all_thetas)
    metric_scaler = StandardScaler().fit(all_metrics[..., metric_idxs].reshape(-1, len(metric_idxs)))
    history_scaler = None
    if history_length > 0:
        some_state_history = np.array(dset_f["initial_trajectories"][0:50])
        history_scaler = MinMaxScaler().fit(some_state_history.reshape((-1, some_state_history.shape[-1])))
    dset_f.close()
    return gain_scaler, theta_scaler, metric_scaler, history_scaler


def get_sysid_history_dataset(cfg, intrinsic_bounds, in_bounds=True, batched=True):
    dset_f = h5py.File(cfg.path_to_dataset, 'r')
    all_gains = np.array(dset_f["gains"])
    gain_dim = all_gains.shape[-1]
    all_metrics = np.array(dset_f["metrics"])
    all_metrics[..., cfg.metric_idxs_to_invert] = 1/(1 + all_metrics[..., cfg.metric_idxs_to_invert])
    all_thetas = np.array(dset_f["intrinsics"])
    batch_size = all_gains.shape[1]
    if "reference_tracks_enc" in dset_f.keys():
        print(f"downsample factor is {cfg.ref_track_ds_factor}")
        print("Using encoded lookahead tracks!")
        all_ref_tracks = np.array(dset_f["reference_tracks_enc"])
        if not cfg.ref_track_per_term_scaling:
            ref_track_enc_scaler = MinMaxScaler(clip=True).fit(all_ref_tracks.reshape(-1, all_ref_tracks.shape[-1]))
            # ref_track_enc_scaler = RobustScaler().fit(all_ref_tracks.reshape(-1, all_ref_tracks.shape[-1]))
        else:
            ref_track_enc_scaler = MinMaxScaler(clip=True).fit(all_ref_tracks.reshape(-1, 1))
            # ref_track_enc_scaler = RobustScaler().fit(all_ref_tracks.reshape(-1, 1))
        ref_track_enc_key = "reference_tracks_enc"

    # else:
    #     ref_track_key = None
    #     all_ref_tracks = np.array(dset_f["reference_tracks"])
    # this is not a sustainable solution as the ref tracks can contain multiple dimensions...
    # ref_track_scaler = MinMaxScaler(clip=True).fit(all_ref_tracks.reshape(-1, all_ref_tracks.shape[-2] * all_ref_tracks.shape[-1]))
    # ref_track_key = "reference_tracks"
    else:
        ref_track_enc_key = None
    theta_dim = all_thetas.shape[-1]

    if all_gains.shape[-1] > 0:
        gain_scaler = MinMaxScaler().fit(all_gains.reshape(-1, all_gains.shape[-1]))
    if all_thetas.shape[-1] > 0:
        theta_scaler = StandardScaler().fit(all_thetas)
    metric_scaler = StandardScaler().fit(all_metrics[..., cfg.metric_idxs].reshape(-1, len(cfg.metric_idxs)))
    # metric_scaler = RobustScaler().fit(all_metrics[..., cfg.metric_idxs].reshape(-1, len(cfg.metric_idxs)))
    # metric_scaler = StandardScaler().fit(1/(all_metrics[..., metric_idxs] + 1e-3).reshape(-1, len(metric_idxs)))
    history_scaler = None
    if cfg.history_length > 0:
        some_state_history = np.array(dset_f["initial_trajectories"][0:1000])
        if cfg.phase_2_method == "student":
            some_state_history[...,cfg.history_sub_idxs] -= some_state_history[...,0:1,cfg.history_sub_idxs]
        some_state_history = some_state_history.reshape((-1, some_state_history.shape[-1]))
        history_scaler = MinMaxScaler().fit(some_state_history)
        # history_scaler = StandardScaler().fit(some_state_history)

    theta_noises = np.random.randn(*all_thetas.shape) * cfg.theta_noise_std
    metric_noises = np.random.randn(*all_metrics.shape) * cfg.metric_noise_std
    if cfg.history_length > 0:
        shape = dset_f["initial_trajectories"].shape
        history_noises = np.random.randn(*shape) * cfg.history_noise_std
        history_noises = history_noises.reshape((shape[0], -1))

    if all_thetas.shape[-1] > 0:
        def theta_pp_fn(theta, idx):
            theta_scaled = theta_scaler.transform(theta)
            theta_scaled += np.random.randn(*theta_scaled.shape) * cfg.theta_noise_std
            # theta_scaled += theta_noises[idx]
            return theta_scaled

        def theta_pp_fn_val(theta, idx):
            theta_scaled = theta_scaler.transform(theta)
            # theta_scaled += theta_noises[idx]
            return theta_scaled
    else:
        def theta_pp_fn(theta, idx):
            return theta

    def metric_pp_fn(metric, idx):
        metric_new = np.array(metric)
        metric_new[...,cfg.metric_idxs_to_invert] = 1/(1+metric[...,cfg.metric_idxs_to_invert])
        metric_scaled = metric_scaler.transform(metric_new)
        metric_scaled += np.random.randn(*metric_scaled.shape) * cfg.metric_noise_std
        return metric_scaled

    def metric_pp_fn_gt(metric, idx):
        metric_new = np.array(metric)
        metric_new[...,cfg.metric_idxs_to_invert] = 1/(1+metric[...,cfg.metric_idxs_to_invert])
        metric_scaled = metric_scaler.transform(metric_new)
        metric_scaled += np.random.randn(*metric_scaled.shape) * cfg.metric_noise_std
        return metric_scaled

    # Do I need to add noise to this?
    if all_gains.shape[-1] > 0:
        def gain_pp_fn(gain, idx):
            return gain_scaler.transform(gain)
    else:
        def gain_pp_fn(gain, idx):
            return gain

    def ref_track_enc_pp_fn(track, idx):
        return ref_track_enc_scaler.transform(track)

    def ref_track_enc_pp_fn_2(track, idx):
        track_flattened = np.reshape(track, (-1, 1))
        track_scaled = ref_track_enc_scaler.transform(track_flattened).reshape(track.shape)
        return track_scaled[...,::cfg.ref_track_ds_factor]

    # def ref_track_pp_fn(track, idx):
    #     track_rs = track.reshape((track.shape[0], -1))
    #     print(f"track_rs shape is {track_rs.shape}")
    #     return ref_track_scaler.transform(track_rs)


    # include fixed noise to have a consistent validation set?
    # traj is shape (64, history_length, 6)
    def history_pp_fn(traj, idx):
        traj = traj[..., -cfg.history_length:, :]
        traj_rs = traj.reshape(-1, traj.shape[-1])
        traj_scaled = history_scaler.transform(traj_rs)
        traj_flat = traj_scaled.reshape(traj.shape[0], -1)
        return traj_flat + np.random.randn(*traj_flat.shape) * cfg.history_noise_std

    def history_pp_fn_gt(traj, idx):
        traj = traj[..., -cfg.history_length:, :]
        traj_rs = traj.reshape(-1, traj.shape[-1])
        traj_scaled = history_scaler.transform(traj_rs)
        traj_flat = traj_scaled.reshape(traj.shape[0], -1)
        return traj_flat

    def history_pp_fn_fixed_noise(traj, idx):
        traj = traj[..., -cfg.history_length:, :]
        traj_rs = traj.reshape(-1, traj.shape[-1])
        traj_scaled = history_scaler.transform(traj_rs)
        traj_flat = traj_scaled.reshape(traj.shape[0], -1)
        return traj_flat + history_noises[idx]


    # potentially don't need metric scaler here.
    # scalers = {"intrinsics":theta_scaler, "gains":gain_scaler, "metrics":metric_scaler}
    transforms_train = {"intrinsics": theta_pp_fn,
                        "gains": gain_pp_fn,
                        "metrics": metric_pp_fn,
                        "metrics_gt": metric_pp_fn_gt,
                        "initial_trajectories": history_pp_fn,
                        "reference_tracks_enc":ref_track_enc_pp_fn}

    transforms_val = {"intrinsics": theta_pp_fn_val,
                      "gains": gain_pp_fn,
                      "metrics": metric_pp_fn_gt,
                      "metrics_gt": metric_pp_fn_gt,
                      "initial_trajectories": history_pp_fn_gt,
                      "reference_tracks_enc":ref_track_enc_pp_fn}

    if cfg.ref_track_per_term_scaling:
        transforms_val["reference_tracks_enc"] = ref_track_enc_pp_fn_2
        transforms_train["reference_tracks_enc"] = ref_track_enc_pp_fn_2

    # if val_history_noise is None:
    #     transforms_val["initial_trajectories"] = history_pp_fn_gt
    # elif val_history_noise == "fixed":
    #     transforms_val["initial_trajectories"] = history_pp_fn_fixed_noise
    # else:
    #     transforms_val["initial_trajectories"] = history_pp_fn

    # need to do train/val split?
    if cfg.history_length > 0:
        history_key = "initial_trajectories"
    else:
        history_key = None

    # get indexes within the bounds
    if in_bounds:
        ib_idxs = get_idxs_in_bounds(all_thetas, intrinsic_bounds[cfg.thetas_randomized])
    else:
        ib_idxs = get_idxs_out_of_bounds(all_thetas, intrinsic_bounds[cfg.thetas_randomized])
    print(f"Using {int(len(ib_idxs) * cfg.train_amt)} batches for training")
    lim = int(cfg.train_amt * ib_idxs.shape[0])
    val_amt = int(0.95 * ib_idxs.shape[0])

    if batched:
        train_dataset = H5IntrinsicBatchedDataset(cfg.path_to_dataset,
                                                  "intrinsics",
                                                  "gains",
                                                  transforms_train,
                                                  ib_idxs[:lim],
                                                  metric_idxs=cfg.metric_idxs,
                                                  metric_key="metrics",
                                                  gt_metric_key="metrics_gt",
                                                  history_key=history_key,
                                                  ref_traj_key=ref_track_enc_key)

        val_dataset = H5IntrinsicBatchedDataset(cfg.path_to_dataset,
                                                "intrinsics",
                                                "gains",
                                                transforms_val,
                                                ib_idxs[val_amt:],
                                                metric_idxs=cfg.metric_idxs,
                                                metric_key="metrics",
                                                gt_metric_key="metrics_gt",
                                                history_key=history_key,
                                                ref_traj_key=ref_track_enc_key)

        train_task_dataloader = DataLoader(train_dataset, batch_size=None, num_workers=0, shuffle=True)
        val_task_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=0, shuffle=True)
    else:
        full_idxs = []
        for idx in ib_idxs:
            full_idxs.append(np.arange(batch_size*idx, batch_size*(idx+1), dtype=int))
        full_idxs = np.array(full_idxs).ravel()
        np.random.shuffle(full_idxs)
        train_idxs = full_idxs[:int(cfg.train_amt*len(full_idxs))]
        val_idxs = full_idxs[int(cfg.train_amt*len(full_idxs)):]
        train_dataset = H5NonBatchedDataset(cfg.path_to_dataset,
                                            "intrinsics",
                                            "gains",
                                            transforms_train,
                                            train_idxs,
                                            metric_idxs=cfg.metric_idxs,
                                            metric_key="metrics",
                                            gt_metric_key="metrics_gt",
                                            history_key=history_key,
                                            ref_traj_key=ref_track_enc_key)

        val_dataset = H5NonBatchedDataset(cfg.path_to_dataset,
                                          "intrinsics",
                                          "gains",
                                          transforms_val,
                                          val_idxs,
                                          metric_idxs=cfg.metric_idxs,
                                          metric_key="metrics",
                                          gt_metric_key="metrics_gt",
                                          history_key=history_key,
                                          ref_traj_key=ref_track_enc_key)

        train_task_dataloader = DataLoader(train_dataset, batch_size=cfg.pretrain_batch_size, num_workers=10, shuffle=True)
        val_task_dataloader = DataLoader(val_dataset, batch_size=cfg.pretrain_batch_size, num_workers=10, shuffle=True)

    return train_dataset, train_task_dataloader, val_dataset, val_task_dataloader
