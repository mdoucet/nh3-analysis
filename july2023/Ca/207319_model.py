
import numpy
import os
from refl1d.names import *
from math import *
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
warnings.simplefilter('ignore', UserWarning)

# Maximum Q-value ##############################################################
q_min = 0.0
q_max = 1.0

reduced_file = "REFL_207319_combined_data_auto.txt"

Q, R, dR, dQ = numpy.loadtxt(reduced_file).T
i_min = min([i for i in range(len(Q)) if Q[i]>q_min])
i_max = max([i for i in range(len(Q)) if Q[i]<q_max])+1

# SNS data is FWHM
dQ_std = dQ/2.35
probe = QProbe(Q[i_min:i_max], dQ_std[i_min:i_max], data=(R[i_min:i_max], dR[i_min:i_max]))

# Materials ####################################################################
Si = SLD(name='Si', rho=2.07, irho=0.0)
THF = SLD(name='THF', rho=5.662, irho=0.0)
Ti = SLD(name='Ti', rho=-3.072, irho=0.0)
Cu = SLD(name='Cu', rho=6.486, irho=0.0)
material = SLD(name='material', rho=2.111, irho=0.0)
SEI = SLD(name='SEI', rho=3.464, irho=0.0)


# Film definition ##############################################################
sample = (  THF(0, 17.5) | SEI(71.49, 12.06) | material(46.25, 14.74) | Cu(567.5, 9.287) | Ti(51.75, 5.954) | Si )

sample['Ti'].thickness.range(10.0, 100.0)
sample['Ti'].material.rho.range(-5.0, 4.0)
sample['Ti'].interface.range(1.0, 15.0)
sample['Cu'].thickness.range(10.0, 800.0)
sample['Cu'].interface.range(1.0, 15.0)
sample['material'].thickness.range(10.0, 300.0)
sample['material'].material.rho.range(1.0, 6.0)
sample['material'].interface.range(1.0, 35.0)
sample['SEI'].thickness.range(10.0, 300.0)
sample['SEI'].material.rho.range(1.0, 6.0)
sample['SEI'].interface.range(5.0, 35.0)



probe.intensity=Parameter(value=1.041,name='normalization')
probe.background=Parameter(value=0.0,name='background')
sample['THF'].interface.range(1.0, 77.0)

################################################################################

expt = Experiment(probe=probe, sample=sample)
problem = FitProblem(expt)
