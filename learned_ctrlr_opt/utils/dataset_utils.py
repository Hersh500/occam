import os
import glob
from typing import Union

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
import h5py


def get_data_from_folders(glob_folder, data_prefixes, folders, num_lim=np.inf):
    folders = [os.path.join(glob_folder, f) for f in folders]
    all_datas = []
    total_num = 0
    num_per_folder = []
    mult_per_folder = []
    shape_per_prefixs = []
    for i, folder in enumerate(folders):
        tmp_prefix = data_prefixes[i][0]
        length = len(os.path.join(folder, tmp_prefix))
        numbers = [int(f[length:-4]) for f in glob.glob(os.path.join(folder, tmp_prefix+"*"))]
        num_per_folder.append(max(numbers)+1)
        mult_per_folder.append(np.load(os.path.join(folder, tmp_prefix+"0.npy")).shape[0])
        if i == 0:
            for pf in data_prefixes[i]:  # assuming that the secondary shapes are the same after the initial dimension
                shape_per_prefixs.append(np.load(os.path.join(folder, pf+"0.npy")).shape)
        total_num += max(numbers)

    full_size = sum([min(num_lim, num_per_folder[j]) * mult_per_folder[j] for j in range(len(folders))])
    all_datas = [np.zeros((full_size, *shape_per_prefixs[i][1:])) for i in range(len(data_prefixes[0]))]

    # for j in range(len(folders)):
    #     for i in range(len(data_prefixes[j])):
    #         all_datas.append(np.zeros((full_size, *shape_per_prefixs[j][i][1:])))

    shift_fwd = 0
    for i, folder in enumerate(folders):
        for k, prefix in enumerate(data_prefixes[i]):
            for j in range(min(num_lim, num_per_folder[i])):
                fname = os.path.join(folder, prefix+f"{j}.npy")
                thing = np.load(fname)
                # all_datas[k][(j+shift_fwd)*mult_per_folder[i]:(j+1+shift_fwd)*mult_per_folder[i],...] = thing
                all_datas[k][j*mult_per_folder[i]+shift_fwd:(j+1)*mult_per_folder[i]+shift_fwd,...] = thing
        shift_fwd += num_per_folder[i] * mult_per_folder[i]
    return all_datas


def print_all_model_checkpoints(folder):
    files = glob.glob(folder+"/*.pt")
    for f in files:
        d = torch.load(f)
        print(f"{f}: {d['val_loss']}")

class CtrlrOptDataset(TensorDataset):
    def __init__(self, gains: Union[np.ndarray, torch.Tensor],
                 metrics: Union[np.ndarray, torch.Tensor],
                 metadata,
                 gains_to_optimize,
                 scaler,
                 bounds,
                 *extrinsics):

        self.scaler = scaler
        self.gains_to_optimize = gains_to_optimize
        self.bounds = bounds
        # Convert to tensors
        if isinstance(gains, np.ndarray):
            self.gains = torch.from_numpy(gains).double()
        else:
            self.gains = gains

        if isinstance(metrics, np.ndarray):
            self.metrics = torch.from_numpy(metrics).double()
        else:
            self.metrics = metrics

        self.metadata = metadata
        extrinsic_list = []
        for e in extrinsics:
            if isinstance(e, np.ndarray):
                extrinsic_list.append(torch.from_numpy(e).double())
            else:
                print(type(e))
        # concat the extrinsics into a single extrinsics tensor
        if len(extrinsic_list) > 0:
            self.extrinsics = torch.cat(extrinsic_list, dim=1)
            super().__init__(torch.cat([self.gains, self.extrinsics], dim=1), self.metrics)
        else:
            self.extrinsics = []
            super().__init__(self.gains, self.metrics)

    def input_size(self):
        x, y = self.__getitem__(0)
        return x.size(-1)

    def output_size(self):
        x, y = self.__getitem__(0)
        return y.size(-1)


# applies pp_fn to every point that is loaded from the file
class HDF5Dataset(TensorDataset):
    def __init__(self, hdf5_file,
                 keys,
                 pp_fns):
        self.hdf5_file = hdf5_file
        self.keys = keys
        self.pp_fns = pp_fns  # applies some preprocessing to trajectories.
        super().__init__()

    def __len__(self):
        return (self.nums[1] - self.nums[0]) * self.num_per

    def __getitem__(self, item):
        # calculate num of item, check if that dataset is in the cache
        item_num = int(item/self.num_per) + self.nums[0]
        if item_num > self.nums[1]:
            raise IndexError(f"trying to access object in file {item_num}, but max possible is {self.nums[1]}")
        sub_idx = item % self.num_per
        vals = []
        for i, pf in enumerate(self.prefixes):
            vals.append(self.pp_fns[i](np.load(os.path.join(self.folder, pf+f"_{item_num}.npy"))[sub_idx]))
        return vals

# Convert a series of numpy files into a single hdf5 group with multiple datasets within it.
def convert_np_arrays_to_hdf5(glob_folder, data_prefixes, folders, num_lim=np.inf):
    folders = [os.path.join(glob_folder, f) for f in folders]
    total_num = 0
    num_per_folder = []
    mult_per_folder = []
    shape_per_prefixs = []
    for i, folder in enumerate(folders):
        tmp_prefix = data_prefixes[i][0]
        length = len(os.path.join(folder, tmp_prefix))
        numbers = [int(f[length:-4]) for f in glob.glob(os.path.join(folder, tmp_prefix+"*"))]
        num_per_folder.append(max(numbers)+1)
        mult_per_folder.append(np.load(os.path.join(folder, tmp_prefix+"0.npy")).shape[0])
        if i == 0:
            for pf in data_prefixes[i]:  # assuming that the secondary shapes are the same after the initial dimension
                shape_per_prefixs.append(np.load(os.path.join(folder, pf+"0.npy")).shape)
        total_num += max(numbers)

    full_size = sum([min(num_lim, num_per_folder[j]) * mult_per_folder[j] for j in range(len(folders))])
    shift_fwd = 0
    with h5py.File(os.path.join(glob_folder, "test.hdf5"), "w") as f:
        keys = [pf.strip(" _0123456789.") for i, pf in enumerate(data_prefixes[0])]
        for i, key in enumerate(keys):
             f.create_dataset(key, (full_size, *shape_per_prefixs[i]), dtype='float')
        for i, folder in enumerate(folders):
            for k, prefix in enumerate(data_prefixes[i]):
                for j in range(min(num_lim, num_per_folder[i])):
                    thing = np.load(os.path.join(folder, prefix+f"{j}.npy"))
                    f[keys[k]][j*mult_per_folder[i]+shift_fwd:(j+1)*mult_per_folder[i]+shift_fwd,...] = thing
            shift_fwd += num_per_folder[i] * mult_per_folder[i]


class H5NonBatchedDataset(TensorDataset):
    def __init__(self, path_to_dataset,
                 intrinsic_key,
                 gain_key,
                 transforms,
                 idxs,
                 metric_key=None,
                 gt_metric_key=None,
                 metric_idxs=None,
                 ref_traj_key=None,
                 history_key=None):

        super().__init__()
        self.path_to_dataset = path_to_dataset
        self.dataset = None

        self.intrinsic_key = intrinsic_key
        self.gain_key = gain_key
        self.metric_key = metric_key
        self.gt_metric_key = gt_metric_key
        self.ref_traj_key = ref_traj_key
        self.transforms = transforms
        # self.start_end_idxs = start_end_idxs
        self.idxs = idxs
        self.metric_idxs = metric_idxs
        self.history_key = history_key
        assert ((metric_idxs is None) == (metric_key is None))

        with h5py.File(self.path_to_dataset, 'r') as f:
            # self.num_batches = self.idxs.shape[0]
            self.num_datapoints = self.idxs.shape[0]
            self.batch_size = f[gain_key].shape[1]

    def __len__(self):
        # return self.num_batches * self.batch_size
        return self.num_datapoints

    def __getitem__(self, item):
        if self.dataset is None:
            self.dataset = h5py.File(self.path_to_dataset, 'r')
        true_idx = self.idxs[item]
        # which_batch = self.idxs[int(item/self.batch_size)]
        which_batch = int(true_idx/self.batch_size)
        which_idx_in_batch = true_idx % self.batch_size
        # print(f"batch idx = {which_batch}")
        intrinsic = torch.from_numpy(self.transforms[self.intrinsic_key](self.dataset[self.intrinsic_key][which_batch:which_batch+1], item)).squeeze()
        gain = torch.from_numpy(self.transforms[self.gain_key](self.dataset[self.gain_key][which_batch,which_idx_in_batch:which_idx_in_batch+1], item)).squeeze()
        output = [gain, intrinsic]
        if self.metric_key is not None:
            if self.metric_key not in self.transforms.keys():
                output.append(torch.from_numpy(self.dataset[self.metric_key][which_batch,which_idx_in_batch,self.metric_idxs]))
            else:
                output.append(torch.from_numpy(
                    self.transforms[self.metric_key](self.dataset[self.metric_key][which_batch,which_idx_in_batch:which_idx_in_batch+1,self.metric_idxs], item)
                ).squeeze())
        if self.gt_metric_key is not None:  # Assumes that the metrics in the dataset are noiseless
            if self.gt_metric_key not in self.transforms.keys():
                raise KeyError("Need gt_metric_key in transforms to return gt_metrics")
            output.append(torch.from_numpy(
                self.transforms[self.gt_metric_key](self.dataset[self.metric_key][which_batch, which_idx_in_batch:which_idx_in_batch+1, self.metric_idxs], item)
            ).squeeze())
        if self.ref_traj_key is not None:
            output.append(torch.from_numpy(
                self.transforms[self.ref_traj_key](self.dataset[self.ref_traj_key][which_batch,which_idx_in_batch:which_idx_in_batch+1], item)
            ).squeeze())
        if self.history_key is not None:
            output.append(torch.from_numpy(
                self.transforms[self.history_key](self.dataset[self.history_key][which_batch,which_idx_in_batch:which_idx_in_batch+1], item)
            ).squeeze())
        return output


# Issues with HDF5 and Pytorch integration listed here:
# https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/15?fbclid=IwAR2jFrRkKXv4PL9urrZeiHT_a3eEn7eZDWjUaQ-zcLP6BRtMO7e0nMgwlKU
class H5IntrinsicBatchedDataset(TensorDataset):
    def __init__(self, path_to_dataset,
                 intrinsic_key,
                 gain_key,
                 transforms,
                 idxs,
                 metric_key=None,
                 gt_metric_key=None,
                 metric_idxs=None,
                 ref_traj_key=None,
                 history_key=None):

        super().__init__()
        self.path_to_dataset = path_to_dataset
        self.dataset = None

        self.intrinsic_key = intrinsic_key
        self.gain_key = gain_key
        self.metric_key = metric_key
        self.gt_metric_key = gt_metric_key
        self.ref_traj_key = ref_traj_key
        self.transforms = transforms
        # self.start_end_idxs = start_end_idxs
        self.idxs = idxs
        self.metric_idxs = metric_idxs
        self.history_key = history_key
        assert ((metric_idxs is None) == (metric_key is None))

        with h5py.File(self.path_to_dataset, 'r') as f:
            self.num_batches = self.idxs.shape[0]
            self.batch_size = f[gain_key].shape[1]

    def __len__(self):
        return self.num_batches

    def __getitem__(self, item):
        # Doing it this way allows multiprocessing, apparently.
        assert item < self.num_batches
        assert item >= 0
        if self.dataset is None:
            self.dataset = h5py.File(self.path_to_dataset, 'r')
        true_idx = self.idxs[item]
        intrinsic = self.dataset[self.intrinsic_key][true_idx]  # should be of shape (num_thetas)
        intrinsic = np.squeeze(intrinsic)
        intrinsic_batch = torch.from_numpy(self.transforms[self.intrinsic_key](np.resize(intrinsic, (self.batch_size, intrinsic.shape[-1])), item))
        gain_batch = torch.from_numpy(self.transforms[self.gain_key](self.dataset[self.gain_key][true_idx], item))
        batches = [gain_batch, intrinsic_batch]
        if self.metric_key is not None:
            if self.metric_key not in self.transforms.keys():
                batches.append(torch.from_numpy(self.dataset[self.metric_key][true_idx,:,self.metric_idxs]))
            else:
                batches.append(torch.from_numpy(
                    self.transforms[self.metric_key](self.dataset[self.metric_key][true_idx,:,self.metric_idxs], item)
                ))
        if self.gt_metric_key is not None:  # Assumes that the metrics in the dataset are noiseless
            if self.gt_metric_key not in self.transforms.keys():
                raise KeyError("Need gt_metric_key in transforms to return gt_metrics")
            batches.append(torch.from_numpy(
                self.transforms[self.gt_metric_key](self.dataset[self.metric_key][true_idx, :, self.metric_idxs], item)
            ))
        if self.ref_traj_key is not None:
            batches.append(torch.from_numpy(
                self.transforms[self.ref_traj_key](self.dataset[self.ref_traj_key][true_idx], item)
            ))
        if self.history_key is not None:
            batches.append(torch.from_numpy(
                self.transforms[self.history_key](self.dataset[self.history_key][true_idx], item)
            ))
        return batches

    def total_num_datapoints(self):
        return self.num_batches * self.batch_size


def get_idxs_out_of_bounds(x, bounds):
    return np.nonzero(np.all(x > bounds[:,1], axis=1) | np.all(x < bounds[:,0], axis=1))[0]

def get_idxs_in_bounds(x, bounds):
    return np.nonzero(np.all(x <= bounds[:,1], axis=1) & np.all(x >= bounds[:,0], axis=1))[0]
    
def merge_h5_datasets(dataset_paths, merged_dataset_path):
    datasets = []
    for path in dataset_paths:
        datasets.append(h5py.File(path, 'r'))

    total_nums = {}
    nontrivial_lengths = {}
    for key in list(datasets[0].keys()):
        print(f"On Key {key}")
        for i in range(len(datasets)):
            if key not in list(datasets[i].keys()):
                print(f"Key {key} not in {list(datasets[i].keys())}")
                continue
            if key not in total_nums.keys():
                total_nums[key] = datasets[i][key].shape[0]
            else:
                total_nums[key] += datasets[i][key].shape[0]
            if len(datasets[i][key].shape) > 3:
                if key not in nontrivial_lengths.keys():
                    nontrivial_lengths[key] = datasets[i][key].shape
                else:
                    if datasets[i][key].shape[-2] < nontrivial_lengths[key][-2]:
                        nontrivial_lengths[key] = datasets[i][key].shape[-2]

    print(total_nums)
    os.makedirs(merged_dataset_path, exist_ok=True)
    merged_dataset = h5py.File(os.path.join(merged_dataset_path,"dataset.hdf5"), 'w')
    # Assumes batch sizes are the same between datasets
    for key in list(datasets[0].keys()):
        idx = 0
        if key in nontrivial_lengths:
            merged_dataset.create_dataset(key, shape=(total_nums[key], *nontrivial_lengths[key][1:]))
        else:
            merged_dataset.create_dataset(key, shape=(total_nums[key], *datasets[0][key].shape[1:]))
        for i in range(len(datasets)):
            if key in nontrivial_lengths.keys():
                merged_dataset[key][idx:idx+datasets[i][key].shape[0], ...] = np.array(datasets[i][key])[...,-nontrivial_lengths[key][-2]:,:]
            else:
                merged_dataset[key][idx:idx+datasets[i][key].shape[0], ...] = np.array(datasets[i][key])
            idx += datasets[i][key].shape[0]
            print(f"key = {key}, idx = {idx}")

    print(merged_dataset.keys())
    for key in merged_dataset.keys():
        print(f"key {key} shape: {merged_dataset[key].shape}")
    merged_dataset.close()
    for dataset in datasets:
        dataset.close()


def prune_batches_with_zeros(path_to_dataset, keys_to_check, keys_to_exclude=["reference_tracks"]):
    try:
        dset_f = h5py.File(path_to_dataset, 'r+')
        arrs = []
        for key in keys_to_check:
            arrs.append(np.array(dset_f[key]))
        mask = np.array([True for i in range(arrs[0].shape[0])], dtype=np.bool)
        weird_idxs = []
        for batch in range(arrs[0].shape[0]):
            for arr in arrs:
                for point in range(arr[batch].shape[0]):
                    if not np.any(arr[batch,point]):  # have an entirely 0 datapoint
                        mask[batch] = False
                        if batch not in weird_idxs:
                            weird_idxs.append(batch)
        print(f"Masking out {weird_idxs}")
        all_keys = dset_f.keys()
        print(f"all keys: {all_keys}")
        for key in all_keys:
            if key not in keys_to_exclude:
                print(key)
                arr = np.array(dset_f[key])
                arr_masked = arr[mask]   # mask out bad batches
                del dset_f[key]
                dset_f.create_dataset(key, shape=arr_masked.shape)
                dset_f[key][...] = arr_masked
                print(f"New dataset entry for key {key} has shape {dset_f[key].shape}")
    finally:
        dset_f.close()


def normalize(point, all_bounds):
    point_norm = (point - all_bounds[..., 0]) / (all_bounds[..., 1] - all_bounds[..., 0])
    return point_norm


def normalize2(point, all_bounds):
    point_norm = normalize(point, all_bounds)  # normalize to [0, 1]
    point_norm = (point_norm - 0.5) * 2  # normalize to [-1, 1]
    return point_norm


def denormalize(point, all_bounds):
    point_denorm = point * (all_bounds[...,1] - all_bounds[...,0]) + all_bounds[...,0]
    return point_denorm


def denormalize2(point, all_bounds):
    point_denorm = point/2 + 0.5
    point_denorm = denormalize(point_denorm, all_bounds)
    return point_denorm
