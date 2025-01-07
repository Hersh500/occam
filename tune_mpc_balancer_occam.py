import numpy as np
import torch
from omegaconf import OmegaConf
import gymnasium as gym
import os
import h5py
from sklearn.preprocessing import MinMaxScaler

from learned_ctrlr_opt.meta_learning.basis_kf import kalman_step
from learned_ctrlr_opt.eval.eval_utils import load_kf_and_scalers, load_task_scaler
from learned_ctrlr_opt.meta_learning.occam_model import OCCAMModel
from learned_ctrlr_opt.opt.random_search import random_search


# Runs the balancer MPC in series with OCCAM. Much simpler, but on a low-compute platform,
# OCCAM could cause issues by periodically blocking control computation.
def tune_balancer_synchronous(env: gym.Env,
                              max_ground_accel: float,
                              mpc_sampling_period: float,
                              nb_mpc_timesteps: int,
                              stage_input_cost_weight: float,
                              stage_state_cost_weight: float,
                              terminal_cost_weight: float,
                              warm_start: bool,
                              cfg: OmegaConf):

    occam_model = OCCAMModel(cfg,
                             path_header="",
                             task_input=True)

    # copy code from upkie repo to run MPC balancer, run_agent.py
    # 
    # replace with my WheeledInvertedPendulum MPC model
    # run OCCAM every couple of iterations, then replace gains.

    return

# Runs the balancer MPC in parallel with OCCAM, requiring two threads. This way, the potentially
# slow random search step does not block the computation of the controller.
def tune_balancer_asynchronous():
    return