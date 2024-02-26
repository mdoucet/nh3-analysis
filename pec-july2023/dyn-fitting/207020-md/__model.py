
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

reduced_file = "/SNS/users/m2d/__data.txt"

Q, R, dR, dQ = numpy.loadtxt(reduced_file).T
i_min = min([i for i in range(len(Q)) if Q[i]>q_min])
i_max = max([i for i in range(len(Q)) if Q[i]<q_max])+1

# SNS data is FWHM
dQ_std = dQ/2.35
probe = QProbe(Q[i_min:i_max], dQ_std[i_min:i_max], data=(R[i_min:i_max], dR[i_min:i_max]))

# Materials ####################################################################
Si = SLD(name='Si', rho=2.07, irho=0.0)
THF = SLD(name='THF', rho=6.09, irho=0.0)
Ti = SLD(name='Ti', rho=-1.61, irho=0.0)
Cu = SLD(name='Cu', rho=6.5, irho=0.0)
material = SLD(name='material', rho=1.915, irho=0.1)
SEI = SLD(name='SEI', rho=3.666, irho=0.0)


# Film definition ##############################################################
sample = (  THF(0, 90.0) | SEI(117.4, 15.02) | material(54.9, 19.76) | Cu(546.0, 8.675) | Ti(51.34, 10.21) | Si )

sample['Ti'].thickness.range(10.0, 100.0)
sample['Ti'].material.rho.range(-3.0, -1.0)
sample['Ti'].interface.range(1.0, 20.0)
sample['Cu'].thickness.range(10.0, 800.0)
sample['Cu'].material.rho.range(6.5, 8.0)
sample['Cu'].interface.range(1.0, 35.0)
sample['material'].thickness.range(10.0, 444.0)
sample['material'].material.rho.range(-2.0, 6.0)
sample['material'].interface.range(1.0, 35.0)
sample['SEI'].thickness.range(100.0, 250.0)
sample['SEI'].material.rho.range(-2.0, 6.0)
sample['SEI'].interface.range(1.0, 45.0)



probe.intensity=Parameter(value=0.9559,name='normalization')
probe.background=Parameter(value=0.0,name='background')
sample['THF'].material.rho.range(5.0, 7.0)
sample['THF'].interface.range(1.0, 90.0)

################################################################################

expt = Experiment(probe=probe, sample=sample)
problem = FitProblem(expt)
