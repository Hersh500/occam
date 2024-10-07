from learned_ctrlr_opt.eval.mob_lkbo_runner import *
from learned_ctrlr_opt.utils.experiment_utils import save_experiment_data
import multiprocessing as mp
import hydra

@hydra.main(version_base=None, config_path="configs/meta_learn_eval_confs", config_name="branin_ood_params_1")
def main(cfg):
    results = mob_lkbo_runner(cfg, cfg.seeds, True)
    offset = cfg.bo_num_offset
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
