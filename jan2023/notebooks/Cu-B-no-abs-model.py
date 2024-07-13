import sys
import os
import time
import numpy as np
np.random.seed(42)

import json



import importlib
git_dir = os.path.join(os.path.expanduser('~'), 'git', 'nh3-analysis')

from refl1d.names import *
from refl1d import errors


import copper_sample as cu

thf_sld = 6.1240
cu_sld = 6.4962

data_file = os.path.join(git_dir, 'jan2023', 'data', 'reduced', 'REFL_201306_combined_data_auto.txt')
meas = cu.Measurement_CuB(data_file, n_sample=20000, fit_abs=False)
meas.thf_sld = thf_sld
meas.cu_sld = cu_sld

problem = meas.get_problem()