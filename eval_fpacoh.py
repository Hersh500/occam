from learned_ctrlr_opt.eval.runner_registry import fpacoh_runners
from learned_ctrlr_opt.utils.experiment_utils import save_experiment_data
from omegaconf import OmegaConf
import os
import h5py
import numpy as np
import torch.multiprocessing as mp
from meta_bo.models import FPACOH_MAP_GP
from meta_bo.meta_environment import RandomBraninMetaEnv
from meta_bo.domain import ContinuousDomain
from learned_ctrlr_opt.systems.hartmann import *
import hydra

@hydra.main(version_base=None, config_path="configs/meta_learn_eval_confs", config_name="branin_ood_params_1")
def main(cfg):
    kf_checkpoint_dir = cfg.kf_ckpt_dir
    kf_cfg = OmegaConf.load(os.path.join(kf_checkpoint_dir, "config.yaml"))
    dset_f = h5py.File(kf_cfg.path_to_dataset, 'r')
    inputs = np.array(dset_f["gains"])
    metrics = np.array(dset_f["metrics"])
    dset_f.close()
    meta_train_tuples = [(inputs[i], metrics[i]) for i in range(int(metrics.shape[0] * kf_cfg.train_amt))]
    if cfg.robot_name == "meta_branin":
        meta_branin_env = RandomBraninMetaEnv()
        # these hyperparameters are from the F-PACOH codebase for this task
        fpacoh_model = FPACOH_MAP_GP(domain=meta_branin_env.domain,
                                     num_iter_fit=2500,
                                     weight_decay=3e-5,
                                     prior_factor=0.06,
                                     task_batch_size=metrics.shape[1],
                                     feature_dim=5,
                                     mean_nn_layers=(32, 32, 32),
                                     kernel_nn_layers=(32, 32, 32))

    elif cfg.robot_name == "meta_hartmann":
        bounds = HartmannInputs.get_bounds()[kf_cfg.gains_to_optimize]
        domain = ContinuousDomain(l=bounds[:,0], u=bounds[:,1])
        fpacoh_model = FPACOH_MAP_GP(domain=domain,
                                     num_iter_fit=2500,
                                     weight_decay=0.03,
                                     prior_factor=0.23,
                                     task_batch_size=metrics.shape[1],
                                     feature_dim=6,
                                     mean_nn_layers=(32, 32, 32),
                                     kernel_nn_layers=(32, 32, 32))
    else:
        raise NotImplementedError("FPACOH is only implemented for branin currently")
    fpacoh_model.meta_fit(meta_train_tuples)
    offset = cfg.fpacoh_num_offset
    args = [(fpacoh_model, cfg, seed) for i, seed in enumerate(cfg.seeds)]
    # pool = mp.Pool(processes=min(max_num_procs, len(cfg.seeds)))
    # results = pool.starmap(fpacoh_runners[cfg.robot_name.lower()], args)
    results = []
    for i, seed in enumerate(cfg.seeds):
        results.append(fpacoh_runners[cfg.robot_name.lower()](*args[i]))
    robot_name = cfg.robot_name
    test_set = cfg.test_set_name
    model_name = "fpacoh"
    for i, r in enumerate(results):
        if i == 0:
            save_experiment_data(robot_name, model_name, test_set, r[0], other_data=r[1], config=cfg, num=i+offset)
        else:
            save_experiment_data(robot_name, model_name, test_set, r[0], other_data=r[1], config=None, num=i+offset)

if __name__ == "__main__":
    main()
