from dataclasses import dataclass, field
from typing import Union, Optional
import yaml

@dataclass
class BayesOptParams:
    acq_fn: str
    use_priors: bool = False
    beta: float = 0.1
    num_evals: int = 20
    n_initial_points: int = 10
    num_priors_to_use: int = -1  # use all priors in dataset


@dataclass
class SysIdParams:
    sys_id_method: str = "none"
    likelihood_method: str = "mse"
    # Particle filtering params
    num_particles: int = 200
    resample_every: int = 5
    explore_noise: float = 0.1

    # MLE params
    mle_learning_rate: float = 0.01
    mle_num_steps: int = 20


@dataclass
class RobotInfo:
    robot_name: str
    ctrl_params_to_optimize: list
    thetas_to_sweep: list
    randomize_thetas: bool = False
    model_thetas: dict = field(default_factory=dict)
    other_args: dict = field(default_factory=dict)


# Experiment Config Class #
@dataclass
class ExperimentConfig:
    weights: list
    robot_info: Union[RobotInfo, dict]
    sys_id: Union[SysIdParams, dict]
    opt_params: Union[BayesOptParams, dict]
    scaler_file: str
    prior_x0_file: str = ""
    prior_y0_file: str = ""
    logdir: str = ""
    num_experiments: int = 1
    prior_run_to_match: Optional[str] = "none"  # copy num_evals, robot params, robot_info from a prior run
    use_ground_truth_theta: Optional[bool] = False
    test_set_dir: Optional[str] = None
    noise_std: Optional[list] = field(default_factory=list)  # standard deviation of noise added to observations at test time

    def __post_init__(self):
        # Post-process loading from YAML
        if isinstance(self.robot_info, dict):
            self.robot_info = RobotInfo(**self.robot_info)
        if isinstance(self.sys_id, dict):
            self.sys_id = SysIdParams(**self.sys_id)
        if isinstance(self.opt_params, dict):
            self.opt_params = BayesOptParams(**self.opt_params)

    @staticmethod
    def load_from_yaml(yaml_file):
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        try:
            return ExperimentConfig(**config)
        except TypeError as E:
            print(E)
            print("Got error mapping yaml to ExperimentConfig class! Check that all variables are named correctly and there are no extra variables.")

    @staticmethod
    def get_dict_from_yaml(yaml_file):
        with open(yaml_file, 'r') as f:
            return yaml.safe_load(f)