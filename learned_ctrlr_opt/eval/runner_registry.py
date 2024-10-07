from learned_ctrlr_opt.eval.tdc_rf_reptile_runner import *
from learned_ctrlr_opt.eval.tdc_rf_kf_runner import *
from learned_ctrlr_opt.eval.tdc_rf_nom_gain_runner import *
from learned_ctrlr_opt.eval.tdc_rf_lkbo_runner import *
from learned_ctrlr_opt.eval.tdc_lifelong_kf_runner import *
from learned_ctrlr_opt.eval.tdc_lifelong_nom_gain_runner import *

from learned_ctrlr_opt.eval.branin_fpacoh_runner import *
from learned_ctrlr_opt.eval.branin_kf_runner import *
from learned_ctrlr_opt.eval.branin_reptile_runner import *

from learned_ctrlr_opt.eval.cf_rf_kf_runner import *
from learned_ctrlr_opt.eval.cf_rf_nom_gain_runner import *
from learned_ctrlr_opt.eval.cf_rf_reptile_runner import *
from learned_ctrlr_opt.eval.cf_rf_lkbo_runner import *

from learned_ctrlr_opt.eval.hartmann_kf_runner import *
from learned_ctrlr_opt.eval.hartmann_reptile_runner import *
from learned_ctrlr_opt.eval.hartmann_fpacoh_runner import *
from learned_ctrlr_opt.eval.hartmann_lkbo_runner import *

from learned_ctrlr_opt.eval.branin_lkbo_runner import *

from learned_ctrlr_opt.eval.cf_rf_ts_runner import *
from learned_ctrlr_opt.eval.tdc_rf_ts_runner import *

ts_runners = {"crazyflie": cf_rf_ts_runner,
              "topdowncar": tdc_rf_ts_runner}

reptile_runners = {"meta_branin": branin_reptile_runner,
                   "topdowncar": tdc_rf_reptile_runner,
                   "crazyflie": cf_rf_reptile_runner,
                   "meta_hartmann": hartmann_reptile_runner}

kf_runners = {"meta_branin": branin_kf_runner,
              "topdowncar": tdc_rf_kf_runner,
              "crazyflie": cf_rf_kf_runner,
              "meta_hartmann": hartmann_kf_runner,
              "topdowncar_ll": tdc_lifelong_kf_runner}

nom_gain_runners = {"topdowncar": tdc_rf_nom_gain_runner,
                    "crazyflie": cf_rf_nom_gain_runner,
                    "topdowncar_ll": tdc_lifelong_nom_gain_runner}


fpacoh_runners = {"meta_branin": branin_fpacoh_runner,
                  "meta_hartmann": hartmann_fpacoh_runner}

lkbo_runners = {"meta_branin": branin_lkbo_runner,
                "meta_hartmann": hartmann_lkbo_runner,
                "crazyflie": cf_rf_lkbo_runner,
                "topdowncar": tdc_rf_lkbo_runner}
