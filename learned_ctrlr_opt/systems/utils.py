from learned_ctrlr_opt.systems.mob_locomotion import Go1HiddenParams, Go1HiddenParamsTest, Go1HiddenParamsTrain
from learned_ctrlr_opt.systems.car_dynamics import CarParams, CarParamsTrain, CarParamsTest
from learned_ctrlr_opt.systems.quadrotor_geom import CrazyFlieParams, CrazyFlieParamsTest, CrazyFlieParamsTrain
from learned_ctrlr_opt.systems.branin import *
from learned_ctrlr_opt.systems.hartmann import *

param_types = {"topdowncar": CarParams,
               "topdowncar_train":CarParamsTrain,
               "topdowncar_test": CarParamsTest,
               "mob_loco": Go1HiddenParams,
               "mob_loco_train": Go1HiddenParamsTrain,
               "mob_loco_test": Go1HiddenParamsTest,
               "quad_geom": CrazyFlieParams,
               "quad_geom_train": CrazyFlieParamsTrain,
               "quad_geom_test": CrazyFlieParamsTest,
               "branin": BraninFnParams,
               "branin_train": BraninFnParamsTrain,
               "branin_test": BraninFnParamsTest,
               "hartmann": HartmannFnParams,
               "hartmann_train": HartmannFnParamsTrain,
               "hartmann_test": HartmannFnParamsTest
               }
