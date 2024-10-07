from learned_ctrlr_opt.eval.runner_registry import kf_runners
from learned_ctrlr_opt.utils.experiment_utils import save_experiment_data
import multiprocessing as mp
import hydra

@hydra.main(version_base=None, config_path="configs/meta_learn_eval_confs", config_name="tdc_ood_tasks_4")
def main(cfg):
    if cfg.no_meta:
        print("Using non-meta-learned KF model!")
        cfg.kf_ckpt_dir = cfg.no_meta_ckpt_dir
        cfg.kf_ckpt_file = cfg.no_meta_ckpt_file
        cfg.kf_num_offset = cfg.no_meta_num_offset
    max_num_procs = 5
    offset = cfg.kf_num_offset
    args = [(cfg, seed, False) for i, seed in enumerate(cfg.seeds)]
    pool = mp.Pool(processes=min(max_num_procs, len(cfg.seeds)))
    results = pool.starmap(kf_runners[cfg.robot_name.lower()], args)
    robot_name = cfg.robot_name
    test_set = cfg.test_set_name
    model_name = "kf_noadapt_" + cfg.kf_ckpt_dir.split('/')[-2]
    for i, r in enumerate(results):
        if i == 0:
            save_experiment_data(robot_name, model_name, test_set, r[0], other_data=r[1], config=cfg, num=i+offset)
        else:
            save_experiment_data(robot_name, model_name, test_set, r[0], other_data=r[1], config=None, num=i+offset)

if __name__ == "__main__":
    main()
