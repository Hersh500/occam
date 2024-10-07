import numpy as np
import h5py
import os
from datetime import datetime
import multiprocessing as mp

from learned_ctrlr_opt.systems.robots import TopDownCar, TopDownCarRandomStartingState
from learned_ctrlr_opt.systems.car_dynamics import CarParamsTrain, CarParams
from learned_ctrlr_opt.systems.car_controller import CarControllerParams
from learned_ctrlr_opt.systems.quadrotor_geom import QuadrotorParams, \
    QuadrotorSE3Control, SE3ControlGains, \
    QuadrotorSE3Control_RandomStart, sample_trajectory_static, CrazyFlieSE3Control, generate_random_circle_traj

track_horizon = 75
max_time = 1500

quadrotor_ref_traj_horizon = (8 * 100) / 5
quadrotor_max_tsteps = 8 * 100


def quadrotor_geom_worker(traj, intrinsics, gains, gain_idxs):
    robot = QuadrotorSE3Control(params=QuadrotorParams(*intrinsics),
                                gains_to_optimize=gain_idxs,
                                trajectory_obj=traj,
                                t_f=8)
    metrics, traj = robot.evaluate_x_return_traj_stochastic(gains)
    track = robot.sample_trajectory(dt=0.05)
    return metrics, traj, track


def gather_batched_quadrotor_sysid_data(n_datapoints,
                                        batch_size,
                                        num_batches_per_intrinsic,
                                        gains_to_randomize,
                                        thetas_to_randomize,
                                        high_level_folder,
                                        randomize_trajs=True,
                                        traj_obj=None):
    assert (randomize_trajs or (not randomize_trajs and traj_obj is not None))
    assert n_datapoints == int(n_datapoints / (batch_size * num_batches_per_intrinsic)) * (
            batch_size * num_batches_per_intrinsic)

    num_batches = int(n_datapoints / (batch_size * num_batches_per_intrinsic))
    intrinsics = np.zeros((num_batches, len(thetas_to_randomize)))
    for i in range(intrinsics.shape[0]):
        intrinsics[i, :] = QuadrotorParams.generate_random(thetas_to_randomize).get_list()
    gains_to_test = np.zeros((n_datapoints, len(gains_to_randomize)))
    for i in range(gains_to_test.shape[0]):
        gains_to_test[i] = SE3ControlGains.generate_random(gains_to_randomize).get_list()[gains_to_randomize]
    gains_to_test = np.reshape(gains_to_test, (num_batches, batch_size, len(gains_to_randomize)))
    if randomize_trajs:
        raise NotImplementedError
        # traj_seeds = np.random.randint(0, 10000, size=n_datapoints)
    else:
        traj_objs = np.array([traj_obj for i in range(n_datapoints)])

    traj_objs = np.reshape(traj_objs, (num_batches, batch_size))

    subfolder = "quad_geom_traj_dataset_" + datetime.now().strftime("%b_%d_%Y_%H%M") + "/"
    if not os.path.exists(os.path.join(high_level_folder, subfolder)):
        os.makedirs(os.path.join(high_level_folder, subfolder), exist_ok=True)

    with h5py.File(os.path.join(high_level_folder, subfolder, "dataset.hdf5"), "w") as f:
        f.create_dataset("intrinsics", shape=intrinsics.shape)
        f["intrinsics"][...] = intrinsics
        f["intrinsics"].attrs["bounds"] = QuadrotorParams.get_bounds()[thetas_to_randomize]
        f["intrinsics"].attrs["idxs"] = thetas_to_randomize
        f.create_dataset("gains", shape=gains_to_test.shape)
        f["gains"][...] = gains_to_test
        f["gains"].attrs["bounds"] = SE3ControlGains.get_bounds()[gains_to_randomize]
        f["gains"].attrs["idxs"] = gains_to_randomize
        f.create_dataset("trajectories", shape=(num_batches, batch_size, quadrotor_max_tsteps, 8))
        f.create_dataset("metrics", shape=(num_batches, batch_size, len(QuadrotorSE3Control.perf_metric_names())))
        f.create_dataset("reference_tracks", shape=(num_batches, batch_size, quadrotor_ref_traj_horizon, 8))
        if not randomize_trajs:
            f["reference_tracks"].attrs["params"] = [traj_obj.radius[0], traj_obj.radius[1], traj_obj.freq[0]]
        pool = mp.Pool(processes=20)
        for b in range(intrinsics.shape[0]):
            args = [(traj_objs[b][i], intrinsics[b], gains_to_test[b][i], gains_to_randomize) for i in
                    range(batch_size)]
            results = pool.starmap(quadrotor_geom_worker, args)
            for j, r in enumerate(results):
                f["metrics"][b, j, ...] = np.array(r[0])
                f["trajectories"][b, j, ...] = np.array(r[1])
                f["reference_tracks"][b, j, ...] = np.array(r[2])
            print(f"finished batch {b}")


def tdc_worker(seed, intrinsics, gain_idxs, gains):
    # X = np.zeros((1, len(robot.gains_to_optimize)))
    # y = np.zeros((1, len(robot.perf_metric_names())))
    # random_gains = robot.ControllerParamsT.generate_random(robot.gains_to_optimize)
    robot = TopDownCar(seed=int(seed),
                       car_params=CarParams(*intrinsics),
                       gains_to_optimize=gain_idxs,
                       length=track_horizon,
                       max_time=max_time)  # making an assumption here that intrinsics are continuous in indexes.
    metrics, traj = robot.evaluate_x_return_traj(gains)
    track = robot.get_track()
    return metrics, traj, track


def gather_batched_tdc_sysid_data(n_datapoints,
                                  batch_size,
                                  num_batches_per_intrinsic,
                                  gains_to_randomize,
                                  thetas_to_randomize,
                                  high_level_folder,
                                  randomize_tracks=True,
                                  same_track_batches=True,
                                  track_seed=None):
    assert (randomize_tracks or (not randomize_tracks and track_seed is not None))
    assert n_datapoints == int(n_datapoints / (batch_size * num_batches_per_intrinsic)) * (
            batch_size * num_batches_per_intrinsic)

    num_batches = int(n_datapoints / (batch_size * num_batches_per_intrinsic))
    intrinsics = np.zeros((num_batches, len(thetas_to_randomize)))
    for i in range(intrinsics.shape[0]):
        intrinsics[i, :] = CarParams.generate_random(thetas_to_randomize).get_list()[thetas_to_randomize]
    gains_to_test = np.zeros((n_datapoints, len(gains_to_randomize)))
    for i in range(gains_to_test.shape[0]):
        gains_to_test[i] = CarControllerParams.generate_random(gains_to_randomize).get_list()[gains_to_randomize]
    gains_to_test = np.reshape(gains_to_test, (num_batches, batch_size, len(gains_to_randomize)))
    if randomize_tracks:  # need to be on a per batch basis, considering it's an unmodeled effect.
        if not same_track_batches:
            track_seeds = np.random.randint(0, 10000, size=n_datapoints)
        else:
            track_seeds = np.zeros((num_batches, batch_size))
            for i in range(track_seeds.shape[0]):
                track_seeds[i, :] = np.random.randint(0, 10000)
    else:
        track_seeds = np.ones(n_datapoints) * track_seed
    track_seeds = np.reshape(track_seeds, (num_batches, batch_size))

    subfolder = "tdc_traj_dataset_" + datetime.now().strftime("%b_%d_%Y_%H%M") + "/"
    if not os.path.exists(os.path.join(high_level_folder, subfolder)):
        os.makedirs(os.path.join(high_level_folder, subfolder), exist_ok=True)

    with h5py.File(os.path.join(high_level_folder, subfolder, "dataset.hdf5"), "w") as f:
        f.create_dataset("intrinsics", shape=intrinsics.shape)
        f["intrinsics"][...] = intrinsics
        f["intrinsics"].attrs["bounds"] = CarParams.get_bounds()[thetas_to_randomize]
        f["intrinsics"].attrs["idxs"] = thetas_to_randomize
        f.create_dataset("gains", shape=gains_to_test.shape)
        f["gains"][...] = gains_to_test
        f["gains"].attrs["bounds"] = CarControllerParams.get_bounds()[gains_to_randomize]
        f["gains"].attrs["idxs"] = gains_to_randomize
        f.create_dataset("trajectories", shape=(num_batches, batch_size, max_time, 6))
        f.create_dataset("metrics", shape=(num_batches, batch_size, len(TopDownCar.perf_metric_names())))
        # should I save the track seeds?
        f.create_dataset("reference_tracks", shape=(num_batches, batch_size, track_horizon, 2))
        if not randomize_tracks:
            f["reference_tracks"].attrs["seed"] = track_seed
        else:
            f.create_dataset("track_seeds", shape=(num_batches, batch_size))
            f["track_seeds"][...] = track_seeds
        pool = mp.Pool(processes=20)
        for b in range(intrinsics.shape[0]):
            args = [(int(track_seeds[b][i]), intrinsics[b], gains_to_test[b][i], gains_to_randomize) for i in
                    range(batch_size)]
            results = pool.starmap(tdc_worker, args)
            for j, r in enumerate(results):
                f["metrics"][b, j, ...] = np.array(r[0])
                f["trajectories"][b, j, ...] = np.array(r[1])
                f["reference_tracks"][b, j, ...] = np.array(r[2])
            print(f"finished batch {b}")


def tdc_rss_worker(seed, initial_gain, intrinsics, gains, gain_idxs,
                   length, initial_length, initial_sensor_traj_length, max_time):
    # X = np.zeros((1, len(robot.gains_to_optimize)))
    # y = np.zeros((1, len(robot.perf_metric_names())))
    # random_gains = robot.ControllerParamsT.generate_random(robot.gains_to_optimize)
    robot = TopDownCarRandomStartingState(seed=int(seed),
                                          initial_gain=initial_gain,
                                          car_params=CarParams(*intrinsics),
                                          gains_to_optimize=gain_idxs,
                                          length=length,
                                          initial_length=initial_length,
                                          initial_sensor_traj_length=initial_sensor_traj_length,
                                          max_time=max_time)  # making an assumption here that intrinsics are continuous in indexes.
    success, metrics, sensor_traj, initial_traj = robot.evaluate_x(gains)
    track = robot.get_track()
    return success, metrics, sensor_traj, initial_traj, track


def gather_batched_tdc_rss_sysid_data(n_datapoints,
                                      batch_size,
                                      num_batches_per_intrinsic,
                                      gains_to_randomize,
                                      thetas_to_randomize,
                                      high_level_folder,
                                      initial_length,
                                      initial_sensor_traj_length,
                                      randomize_tracks=True,
                                      same_track_batches=False,  # keep track same for all points in a batch
                                      track_seed=None,
                                      max_time=1500,
                                      params_t=CarParamsTrain):
    assert (randomize_tracks or (not randomize_tracks and track_seed is not None))
    assert n_datapoints == int(n_datapoints / (batch_size * num_batches_per_intrinsic)) * (
            batch_size * num_batches_per_intrinsic)

    # Instead of doing this work upfront, I should just generate it dynamically?
    num_batches = int(n_datapoints / (batch_size * num_batches_per_intrinsic))
    intrinsics = np.zeros((num_batches, len(thetas_to_randomize)))
    gains_to_test = np.zeros((num_batches, batch_size, len(gains_to_randomize)))

    # track_seeds = np.zeros((num_batches, batch_size))
    if randomize_tracks:
        if same_track_batches:
            track_seeds = np.random.randint(0, 10000, size=(num_batches, 1))
            track_seeds = np.repeat(track_seeds, batch_size, axis=1)
        else:
            track_seeds = np.random.randint(0, 10000, size=(num_batches, batch_size))
    else:
        track_seeds = np.array([[track_seed] for k in range(num_batches)])
        track_seeds = np.repeat(track_seeds, batch_size, axis=1)

    subfolder = "tdc_rss_traj_dataset_" + datetime.now().strftime("%b_%d_%Y_%H%M") + "/"
    if not os.path.exists(os.path.join(high_level_folder, subfolder)):
        os.makedirs(os.path.join(high_level_folder, subfolder), exist_ok=True)

    with h5py.File(os.path.join(high_level_folder, subfolder, "dataset.hdf5"), "w") as f:
        f.create_dataset("intrinsics", shape=intrinsics.shape)
        f["intrinsics"].attrs["bounds"] = CarParams.get_bounds()[thetas_to_randomize]
        f["intrinsics"].attrs["idxs"] = thetas_to_randomize
        f.create_dataset("gains", shape=gains_to_test.shape)
        f.create_dataset("initial_gains", shape=gains_to_test.shape)
        f["gains"].attrs["bounds"] = CarControllerParams.get_bounds()[gains_to_randomize]
        f["gains"].attrs["idxs"] = gains_to_randomize
        f.create_dataset("trajectories", shape=(num_batches, batch_size, max_time, 6))
        f.create_dataset("initial_trajectories", shape=(num_batches, batch_size, initial_sensor_traj_length, 6))
        f["initial_trajectories"].attrs["initial_length"] = initial_length
        f.create_dataset("metrics", shape=(num_batches, batch_size, len(TopDownCar.perf_metric_names())))
        # does h5py store something in memory? Why is it that when I generate more data, the computer freezes?
        if not randomize_tracks:
            f.create_dataset("reference_tracks", shape=(track_horizon, 2))
            f["reference_tracks"].attrs["seed"] = track_seed
        else:
            f.create_dataset("reference_tracks", shape=(num_batches, batch_size, track_horizon, 2))
            f.create_dataset("track_seeds", shape=(num_batches, batch_size))
            f["track_seeds"][...] = track_seeds
        num_done = 0
        for b in range(intrinsics.shape[0]):
            num_successfully_done = 0
            batch_intrinsic = params_t.generate_random(thetas_to_randomize).get_list()[thetas_to_randomize]
            f["intrinsics"][b] = batch_intrinsic
            while num_successfully_done < batch_size:
                batch_initial_gains = np.zeros((20, len(gains_to_randomize)))
                batch_gains = np.zeros((20, len(gains_to_randomize)))
                for i in range(batch_gains.shape[0]):
                    batch_gains[i] = CarControllerParams.generate_random(gains_to_randomize).get_list()[
                        gains_to_randomize]
                    batch_initial_gains[i] = CarControllerParams.generate_random(gains_to_randomize).get_list()[
                        gains_to_randomize]
                args = [(track_seeds[b, i],
                         batch_initial_gains[i],
                         batch_intrinsic,
                         batch_gains[i],
                         gains_to_randomize,
                         track_horizon,
                         initial_length,
                         initial_sensor_traj_length,
                         max_time) for i in range(batch_gains.shape[0])]

                pool = mp.Pool(processes=20)
                results = pool.starmap(tdc_rss_worker, args)
                for j, r in enumerate(results):
                    if r[0]:
                        f["gains"][b, num_successfully_done] = batch_gains[j]
                        f["initial_gains"][b, num_successfully_done] = batch_initial_gains[j]
                        f["metrics"][b, num_successfully_done, ...] = np.array(r[1])
                        f["trajectories"][b, num_successfully_done, ...] = np.array(r[2])
                        f["initial_trajectories"][b, num_successfully_done, ...] = np.array(r[3])
                        f["reference_tracks"][b, num_successfully_done, ...] = np.array(r[4])
                        num_successfully_done += 1
                        if num_successfully_done >= batch_size:
                            break
                pool.close()
            print(f"finished batch {b}")


def quadrotor_geom_rss_worker(quad_t, traj, initial_gain,
                              initial_tf, initial_traj_length,
                              intrinsics, gains, gain_idxs, t_f):
    # X = np.zeros((1, len(robot.gains_to_optimize)))
    # y = np.zeros((1, len(robot.perf_metric_names())))
    # random_gains = robot.ControllerParamsT.generate_random(robot.gains_to_optimize)
    robot = quad_t(params=QuadrotorParams(*intrinsics),
                   gains_to_optimize=gain_idxs,
                   trajectory_obj=traj,
                   t_f=t_f,
                   initial_gain=initial_gain,
                   initial_tf=initial_tf,
                   initial_sensor_traj_length=initial_traj_length)
    success, metrics, traj, initial_traj = robot.evaluate_x(gains)
    track = robot.sample_trajectory(dt=0.05)
    return success, metrics, traj, initial_traj, track


def gather_batched_quad_geom_rss_sysid_data(n_datapoints,
                                            batch_size,
                                            num_batches_per_intrinsic,
                                            gains_to_randomize,
                                            thetas_to_randomize,
                                            high_level_folder,
                                            initial_tf,
                                            initial_sensor_traj_length_tsteps,
                                            randomize_tracks=True,
                                            same_track_batches=True,  # keep track same for all points in a batch
                                            traj_obj=None,
                                            t_f=8,
                                            quad_t=QuadrotorSE3Control_RandomStart,
                                            test_set = False):
    num_proc = 20
    assert (randomize_tracks or (not randomize_tracks and traj_obj is not None))
    assert n_datapoints == int(n_datapoints / (batch_size * num_batches_per_intrinsic)) * (
            batch_size * num_batches_per_intrinsic)

    num_batches = int(n_datapoints / (batch_size * num_batches_per_intrinsic))
    intrinsics = np.zeros((num_batches, len(thetas_to_randomize)))
    gains_to_test = np.zeros((num_batches, batch_size, len(gains_to_randomize)))

    subfolder = "quad_geom_rss_traj_dataset_" + datetime.now().strftime("%b_%d_%Y_%H%M") + "/"
    if not os.path.exists(os.path.join(high_level_folder, subfolder)):
        os.makedirs(os.path.join(high_level_folder, subfolder), exist_ok=True)

    with h5py.File(os.path.join(high_level_folder, subfolder, "dataset.hdf5"), "w") as f:
        f.create_dataset("intrinsics", shape=intrinsics.shape)
        f["intrinsics"].attrs["bounds"] = quad_t.ParamsT.get_bounds()[thetas_to_randomize]
        f["intrinsics"].attrs["idxs"] = thetas_to_randomize
        f.create_dataset("gains", shape=gains_to_test.shape)
        f.create_dataset("initial_gains", shape=gains_to_test.shape)
        f["gains"].attrs["bounds"] = quad_t.ControllerParamsT.get_bounds()[gains_to_randomize]
        f["gains"].attrs["idxs"] = gains_to_randomize
        f.create_dataset("trajectories", shape=(num_batches, batch_size, int(t_f / 0.01), 9))
        # f.create_dataset("initial_trajectories", shape=(num_batches, batch_size, initial_sensor_traj_length_tsteps, 12))
        f.create_dataset("initial_trajectories", shape=(num_batches, batch_size, initial_sensor_traj_length_tsteps, 9))
        f["initial_trajectories"].attrs["initial_length_tsteps"] = initial_sensor_traj_length_tsteps
        f.create_dataset("metrics", shape=(num_batches, batch_size, len(QuadrotorSE3Control.perf_metric_names())))
        if not randomize_tracks:
            f.create_dataset("reference_tracks", shape=(int(t_f / 0.05), 8))
            f["reference_tracks"][...] = sample_trajectory_static(traj_obj, t_f, dt=0.05)
            f["reference_tracks"].attrs["traj_params"] = np.array([traj_obj.radius, traj_obj.freq, traj_obj.center])
            traj_objs = [[traj_obj for _ in range(batch_size)] for tmp in range(num_batches)]
        else:
            traj_objs = []
            all_traj_params = np.zeros((num_batches, batch_size, 3, 3))
            for tmp in range(num_batches):
                traj_objs_sublist = []
                traj_obj, traj_params = generate_random_circle_traj(0.7)
                for tmp2 in range(batch_size):
                    if not same_track_batches:
                        traj_obj, traj_params = generate_random_circle_traj(0.7)
                    traj_objs_sublist.append(traj_obj)
                    all_traj_params[tmp, tmp2] = traj_params
                traj_objs.append(traj_objs_sublist)
            f.create_dataset("reference_tracks", shape=all_traj_params.shape)
            f["reference_tracks"][...] = all_traj_params
        num_batches_done = 0
        while num_batches_done < intrinsics.shape[0]:
            b = num_batches_done
            num_successfully_done = 0
            if not test_set:
                batch_intrinsic = quad_t.TrainParamsT.generate_random(thetas_to_randomize).get_list()[thetas_to_randomize]
            else:
                batch_intrinsic = quad_t.TestParamsT.generate_random(thetas_to_randomize).get_list()[thetas_to_randomize]
            num_tried = 0
            while num_successfully_done < batch_size:
                batch_initial_gains = np.zeros((num_proc, len(gains_to_randomize)))
                batch_gains = np.zeros((num_proc, len(gains_to_randomize)))
                for i in range(batch_gains.shape[0]):
                    batch_gains[i] = quad_t.ControllerParamsT.generate_random(gains_to_randomize).get_list()[
                        gains_to_randomize]
                    batch_initial_gains[i] = quad_t.ControllerParamsT.generate_random(gains_to_randomize).get_list()[
                        gains_to_randomize]
                args = [(quad_t,
                         traj_objs[num_batches_done][i],
                         batch_initial_gains[i],
                         initial_tf,
                         initial_sensor_traj_length_tsteps,
                         batch_intrinsic,
                         batch_gains[i],
                         gains_to_randomize,
                         t_f) for i in range(batch_gains.shape[0])]
                pool = mp.Pool(processes=num_proc)
                results = pool.starmap(quadrotor_geom_rss_worker, args)
                pool.close()
                for j, r in enumerate(results):
                    if r[0]:
                        f["gains"][b, num_successfully_done] = batch_gains[j]
                        f["initial_gains"][b, num_successfully_done] = batch_initial_gains[j]
                        f["metrics"][b, num_successfully_done, ...] = np.array(r[1])
                        f["trajectories"][b, num_successfully_done, ...] = np.array(r[2])
                        f["initial_trajectories"][b, num_successfully_done, ...] = np.array(r[3])
                        num_successfully_done += 1
                        if num_successfully_done >= batch_size:
                            break
                num_tried += 1
                # If we're stuck on this intrinsic for too long, just move on to another.
                if num_tried > 10:
                    print(f"Skipping intrinsic {batch_intrinsic}!")
                    break
            if num_tried <= 10:
                f["intrinsics"][b] = batch_intrinsic
                print(f"finished batch {b}, num batches tried was {num_tried}")
                num_batches_done += 1
