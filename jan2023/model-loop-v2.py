"""
refl1d_cli.py --fit=dream --steps=100 --burn=100 --store=results model-loop.py

Use --batch is you dont' want to see the output and pop up the plots.
"""
import sys
import numpy as np
import os

from refl1d.names import QProbe, Parameter, FitProblem
sys.path.append(os.path.expanduser('~/git/analysis_playground/bayesian-fitting'))
import model_utils

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
warnings.simplefilter('ignore', UserWarning)

# Parse input arguments ########################################################
# First argument is the data file to use
reduced_file = sys.argv[1]

# Second argument is the starting model [experiment description]
expt_file = sys.argv[2]

# Third argument is the error information used for setting the prior
err_file = sys.argv[3]

prior_scale = 0.1

# Load data ####################################################################
q_min = 0.0
q_max = 0.4

try:
    Q, R, dR, dQ = np.loadtxt(reduced_file).T
except:
    Q, R, dR = np.loadtxt(reduced_file).T
    dQ = 0.02*Q

i_min = np.min([i for i in range(len(Q)) if Q[i]>q_min])
i_max = np.max([i for i in range(len(Q)) if Q[i]<q_max])+1

# SNS data is FWHM
dQ_std = dQ/2.35
probe = QProbe(Q[i_min:i_max], dQ_std[i_min:i_max], data=(R[i_min:i_max], dR[i_min:i_max]))

# Experiment ###################################################################
expt = model_utils.expt_from_json_file(expt_file, probe=probe,
                                       model_err_json_file=err_file,
                                       prior_scale=prior_scale, set_ranges=False)

expt.sample['material'].thickness.range(10.0, 300.0)
expt.sample['material'].material.rho.range(-1.0, 10.0)
expt.sample['material'].material.irho.range(0.0, 0.3)
expt.sample['material'].interface.range(1.0, 45.0)
expt.sample['SEI'].thickness.range(10.0, 800.0)
expt.sample['SEI'].material.rho.range(-1.0, 6.0)
expt.sample['SEI'].material.irho.range(0.0, 0.3)
expt.sample['SEI'].interface.range(1.0, 100.0)
expt.sample['THF'].interface.range(1.0, 100.0)

################################################################################
problem = FitProblem(expt)
