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


# 0.1 was used so far (Jan 2023) with good results
#prior_scale = 0.1
prior_scale = 1

# Load data ####################################################################
q_min = 0.0
q_max = 0.4

Q = np.arange(0.008, 1., 0.01)
dQ = Q*0.028

# SNS data is FWHM
dQ_std = dQ/2.35
probe = QProbe(Q, dQ_std, data=None)

# Experiment ###################################################################
expt = model_utils.expt_from_json_file('../dyn-fitting/207282-expt.json', probe=probe,
                                       model_err_json_file='../dyn-fitting/207282-err.json',
                                       prior_scale=prior_scale, set_ranges=False)

#sample['material'].thickness = constraint_thickness(sample['material'].thickness)

expt.sample['Cu'].interface.range(5.0, 22.0)
#expt.sample['Cu'].thickness.range(expt.sample['Cu'].thickness.value*0.97, expt.sample['Cu'].thickness.value*1.03)


expt.sample['material'].thickness.range(10.0, 200.0)
expt.sample['material'].material.rho.range(-1.0, 4.0)
#expt.sample['material'].material.irho.range(0.0, 0.3)
#expt.sample['material'].interface.range(1.0, 22.0)
#expt.sample['material'].interface.range(5, expt.sample['material'].thickness.value/2+5)


expt.sample['SEI'].thickness.range(75.0, 350.0)
expt.sample['SEI'].material.rho.range(-4, 6.3)
#expt.sample['SEI'].material.irho.range(0.0, 0.3)
expt.sample['SEI'].interface.range(8.0, 35.0)

neigh_thick = min(expt.sample['material'].thickness.value, expt.sample['SEI'].thickness.value)

#expt.sample['SEI'].interface.range(5, expt.sample['material'].thickness.value/2+5)
#expt.sample['SEI'].interface.range(5, neigh_thick+5)

def thf_rough(thick):
    return thick/2.0

#expt.sample['THF'].interface.range(1.0, 150.0)
#expt.sample['THF'].interface.range(5, expt.sample['SEI'].thickness.value/2+5)
expt.sample['THF'].interface = thf_rough(expt.sample['SEI'].thickness)

#probe.intensity.range(0.95, 1.8)


################################################################################
problem = FitProblem(expt)
