import numpy as np
import os
from typing import Dict
import pickle
from omegaconf import OmegaConf


EXPERIMENT_RESULTS_FOLDER = "experiment_results"
TEST_SETS_FOLDER = "test_sets"

def sum_multiple_dicts(dicts, keys):
    if isinstance(keys, str):
        keys = [keys]
    new_dict = {}
    for key in keys:
        s = 0
        for i, d in enumerate(dicts):
            s += d[key]
        new_dict[key] = s
    return new_dict

def get_mean_across_multiple_dicts(dicts, keys):
    new_dict = {}
    for key in keys:
        all_arrays = np.zeros((len(dicts), *dicts[0][key].shape))
        for i, d in enumerate(dicts):
            all_arrays[i] = d[key]
        new_dict[key] = np.mean(all_arrays, axis=0)
    return new_dict

def get_vars_across_multiple_dicts(dicts, keys, axes):
    new_dict = {}
    for key in keys:
        all_arrays = np.zeros((len(dicts), *dicts[0][key].shape))
        for i, d in enumerate(dicts):
            all_arrays[i] = d[key]
        new_dict[key] = np.var(all_arrays, axis=axes)
    return new_dict

def get_mean_filter_launch_fails(dicts, keys, trial_axis):
    new_dict_mean = {}
    if isinstance(keys, str):
        keys = [keys]
    metric_key = keys[0]

    all_arrays = np.zeros((len(dicts), *dicts[0][metric_key].shape))
    for i, d in enumerate(dicts):
        all_arrays[i] = d[metric_key]
    all_arrays = np.reshape(all_arrays, (-1, *dicts[0][metric_key].shape[trial_axis:]))
    idxs_to_include = all_arrays[:,0,...] != 0
    idxs_to_include = idxs_to_include.squeeze()

    for key in keys:
        all_arrays = np.zeros((len(dicts), *dicts[0][key].shape))
        for i, d in enumerate(dicts):
            all_arrays[i] = d[key]
        all_arrays_rs = np.reshape(all_arrays, (-1, *dicts[0][key].shape[trial_axis:]))
        new_dict_mean[key] = np.mean(all_arrays_rs[idxs_to_include], axis=0).squeeze()
    return new_dict_mean

def get_mean_std_filter_launch_fails(dicts, keys, trial_axis):
    new_dict_mean = {}
    new_dict_var = {}
    if isinstance(keys, str):
        keys = [keys]
    metric_key = keys[0]
    
    all_arrays = np.zeros((len(dicts), *dicts[0][metric_key].shape))
    for i, d in enumerate(dicts):
        all_arrays[i] = d[metric_key]
    all_arrays = np.reshape(all_arrays, (-1, *dicts[0][metric_key].shape[trial_axis:]))
    idxs_to_include = all_arrays[:,0,...] != 0
    idxs_to_include = idxs_to_include.squeeze()

    for key in keys:
        all_arrays = np.zeros((len(dicts), *dicts[0][key].shape))
        for i, d in enumerate(dicts):
            all_arrays[i] = d[key]
        all_arrays_rs = np.reshape(all_arrays, (-1, *dicts[0][key].shape[trial_axis:]))
        new_dict_mean[key] = np.mean(all_arrays_rs[idxs_to_include], axis=0).squeeze()
        # variance is a little trickier for this case...
        shape = dicts[0][key].shape
        variance_array = np.zeros((shape[0], shape[1], shape[2]))
        for j in range(shape[0]):
            for k in range(shape[1]):
                successful_task_trials = []
                for i, d, in enumerate(dicts):
                    if d[metric_key][j,k,0] != 0:
                        successful_task_trials.append(d[key][j,k])
                if len(successful_task_trials) > 0:
                    variance_array[j, k, :] = np.std(successful_task_trials, axis=0).squeeze()
        new_dict_var[key] = np.mean(variance_array, axis=(0,1))
    return new_dict_mean, new_dict_var, 0


def get_mean_std_filter_all_fails(dicts, keys, trial_axis, mean_only=False):
    new_dict_mean = {}
    new_dict_std = {}
    mttf = 0
    total_failed = 0
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        all_arrays = np.zeros((len(dicts), *dicts[0][key].shape))
        for i, d in enumerate(dicts):
            all_arrays[i] = d[key]
        all_arrays_rs = np.reshape(all_arrays, (-1, *dicts[0][key].shape[trial_axis:]))
#         all_arrays_rs_seeds = np.reshape(all_arrays, (len_dicts, -1, *dicts[0][key].shape[trial_axis:]))
        new_dict_mean[key] = np.zeros((dicts[0][key].shape[trial_axis:]))
        new_dict_std[key] = np.zeros((dicts[0][key].shape[trial_axis:]))
        for t in range(new_dict_mean[key].shape[0]):
            idxs_to_include = all_arrays_rs[:,t,...] != 0
            if t > 0:
                failed_this_iter = np.logical_and(all_arrays_rs[:,t-1,...] != 0,
                                                  all_arrays_rs[:,t,...] == 0)
                mttf += np.count_nonzero(failed_this_iter) * t
                total_failed += np.count_nonzero(failed_this_iter)
            if len(idxs_to_include.squeeze().shape) > 1:
                new_dict_mean[key][t] = np.mean(all_arrays_rs[idxs_to_include.squeeze()[:,0],t], axis=0)
            else:
                new_dict_mean[key][t] = np.mean(all_arrays_rs[idxs_to_include.squeeze(),t], axis=0)
            vals = []
            if not mean_only:
                for traj in range(all_arrays.shape[1]):
                    for param in range(all_arrays.shape[2]):
                        idxs = all_arrays[:,traj,param,t] != 0
                        if np.count_nonzero(idxs) > 0:
                            vals.append(np.std(all_arrays[idxs.squeeze(), traj, param, t]))
                new_dict_std[key][t] = np.mean(vals)
    return new_dict_mean, new_dict_std, mttf/(total_failed+1e-2)

def get_mean_filter_crashed_runs(dicts, keys, trial_axis, mean_only=False):
    new_dict_mean = {}
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        num = 0
        all_arrays = np.zeros((len(dicts), *dicts[0][key].shape))
        for i, d in enumerate(dicts):
            all_arrays[i] = d[key]
        all_arrays_rs = np.reshape(all_arrays, (-1, *dicts[0][key].shape[trial_axis:]))
        #         all_arrays_rs_seeds = np.reshape(all_arrays, (len_dicts, -1, *dicts[0][key].shape[trial_axis:]))
        new_dict_mean[key] = np.zeros((dicts[0][key].shape[trial_axis:]))
        for i in range(all_arrays_rs.shape[0]):
            if np.all(all_arrays_rs[i,...]):
                num += 1
                new_dict_mean[key] += all_arrays_rs[i]
        new_dict_mean[key] /= num
    return new_dict_mean


def uninvert_raw_metrics(dicts, keys, inverted_metric_idxs, trial_axis=2):
    new_dict = {}
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        num = 0
        all_arrays = np.zeros((len(dicts), *dicts[0][key].shape))
        for i, d in enumerate(dicts):
            all_arrays[i] = d[key]
        all_arrays_rs = np.reshape(all_arrays, (-1, *dicts[0][key].shape[trial_axis:]))
        new_dict[key] = np.zeros((dicts[0][key].shape[trial_axis:]))
        for i in range(all_arrays_rs.shape[0]):
            for t in range(all_arrays_rs.shape[1]):
                if np.all(all_arrays_rs[i,t]):
                    all_arrays_rs[i,t,inverted_metric_idxs] = (1-all_arrays_rs[i,t,inverted_metric_idxs])/all_arrays_rs[i,t,inverted_metric_idxs]
        all_arrays_rs = np.reshape(all_arrays_rs, all_arrays.shape)
        new_dict[key] = all_arrays_rs
    return new_dict


def save_experiment_data(robot: str,
                         model_name: str,
                         test_set_name: str,
                         result_arrays: Dict,
                         other_data: Dict = None,
                         config=None,
                         num=None):
    path = os.path.join(EXPERIMENT_RESULTS_FOLDER,robot,test_set_name,model_name)
    os.makedirs(path, exist_ok=True)
    if num is None:
        num = len(os.listdir(path)) + 1
    print(f"This is experiment {num} in this folder")
    np.savez(os.path.join(path, f"exp_{num}_results.npz"), **result_arrays)
    if other_data is not None:
        np.savez(os.path.join(path, f"exp_{num}_other_data.npz"), **other_data)
    if config is not None:
        OmegaConf.save(config, f=os.path.join(path, f"config_{num}.yaml"))


def load_experiment_data(robot: str,
                         model_name: str,
                         num: int,
                         test_set_name: str):
    exp_str = f"exp_{num}_results.npz"
    other_str = f"exp_{num}_other_data.npz"
    results = np.load(os.path.join(EXPERIMENT_RESULTS_FOLDER,robot,test_set_name,model_name,exp_str))
    try:
        other_data = np.load(os.path.join(EXPERIMENT_RESULTS_FOLDER, robot, test_set_name, model_name, other_str))
    except FileNotFoundError:
        other_data = None
    return results, other_data


def save_test_set(robot: str,
                  test_set_name: str,
                  param_arrays,
                  task_arrays=None,
                  robot_kwargs=None):
    path = os.path.join(TEST_SETS_FOLDER, robot, test_set_name)
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "params.npy"), param_arrays)
    if task_arrays is not None:
        np.save(os.path.join(path, "tasks.npy"), task_arrays)
    if robot_kwargs is not None:
        with open(os.path.join(path, "robot_kwargs"), 'wb') as f:
            pickle.dump(robot_kwargs, f)


def load_test_set(robot,
                  test_set_name):
    path = os.path.join(TEST_SETS_FOLDER, robot, test_set_name)
    params = np.load(os.path.join(path, "params.npy"))
    try:
        tasks = np.load(os.path.join(path, "tasks.npy"))
    except FileNotFoundError:
        print("No tasks for this dataset!")
        tasks = None
    try:
        with open(os.path.join(path, "robot_kwargs"), 'rb') as f:
            robot_kwargs = pickle.load(f)
    except FileNotFoundError:
        robot_kwargs = None
    return params, tasks, robot_kwargs
