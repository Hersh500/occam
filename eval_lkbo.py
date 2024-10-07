from learned_ctrlr_opt.eval.runner_registry import lkbo_runners
from learned_ctrlr_opt.utils.experiment_utils import save_experiment_data
import multiprocessing as mp
import hydra

@hydra.main(version_base=None, config_path="configs/meta_learn_eval_confs", config_name="branin_ood_params_1")
def main(cfg):
    max_num_procs = 5
    offset = cfg.bo_num_offset
    args = [(cfg, seed) for i, seed in enumerate(cfg.seeds)]
    pool = mp.Pool(processes=min(max_num_procs, len(cfg.seeds)))
    results = pool.starmap(lkbo_runners[cfg.robot_name.lower()], args)
    robot_name = cfg.robot_name
    test_set = cfg.test_set_name
    model_name = "lkbo"
    for i, r in enumerate(results):
        if i == 0:
            save_experiment_data(robot_name, model_name, test_set, r[0], other_data=r[1], config=cfg, num=i+offset)
        else:
            save_experiment_data(robot_name, model_name, test_set, r[0], other_data=r[1], config=None, num=i+offset)

if __name__ == "__main__":
    main()
