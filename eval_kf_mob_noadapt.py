from learned_ctrlr_opt.eval.mob_kf_runner import *
from learned_ctrlr_opt.utils.experiment_utils import save_experiment_data
import hydra

# This requires a separate script due to CUDA/Isaac Gym/multiprocessing issues
@hydra.main(version_base=None, config_path="configs/meta_learn_eval_confs", config_name="mob_ood_params_1")
def main(cfg):
    results = mob_kf_runner(cfg, cfg.seeds, False)
    offset = cfg.kf_num_offset
    robot_name = cfg.robot_name
    test_set = cfg.test_set_name
    model_name = "kf_noadapt_" + cfg.kf_ckpt_dir.split('/')[-2]

    for i, r in enumerate(results):
        if i == 0:
            save_experiment_data(robot_name, model_name, test_set, r[0], other_data=r[1], config=cfg, num=int(i+offset))
        else:
            save_experiment_data(robot_name, model_name, test_set, r[0], other_data=r[1], config=None, num=(i+offset))


if __name__ == "__main__":
    main()
