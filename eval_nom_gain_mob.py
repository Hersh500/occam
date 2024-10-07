from learned_ctrlr_opt.eval.mob_nom_gain_runner import *
from learned_ctrlr_opt.utils.experiment_utils import *
import hydra

@hydra.main(version_base=None, config_path="configs/meta_learn_eval_confs", config_name="tdc_ood_tasks_4")
def main(cfg):
    offset = cfg.ig_num_offset
    results = mob_nom_gain_runner(cfg, cfg.seeds)
    robot_name = cfg.robot_name
    test_set = cfg.test_set_name
    model_name = "nom_gain"
    for i, r in enumerate(results):
        if i == 0:
            save_experiment_data(robot_name, model_name, test_set, r[0], other_data=r[1], config=cfg, num=i+offset)
        else:
            save_experiment_data(robot_name, model_name, test_set, r[0], other_data=r[1], config=None, num=i+offset)

if __name__ == "__main__":
    main()
