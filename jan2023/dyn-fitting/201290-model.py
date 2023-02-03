
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
THF = SLD(name='THF', rho=6.153, irho=0.0)
SiOx = SLD(name='SiOx', rho=2.653, irho=0.0)
Ti = SLD(name='Ti', rho=-1.981, irho=0.0)
Cu = SLD(name='Cu', rho=6.471, irho=0.0)
material = SLD(name='material', rho=2.754, irho=0.05289)
SEI = SLD(name='SEI', rho=3.202, irho=0.2398)


# Film definition ##############################################################
sample = (  THF(0, 71.86) | SEI(244.5, 12.23) | material(97.74, 18.88) | Cu(530.0, 6.168) | Ti(52.83, 8.655) | SiOx(25.06, 2.648) | Si )

sample['material'].thickness.range(10.0, 300.0)
sample['material'].material.rho.range(-1.0, 10.0)
sample['material'].material.irho.range(0.0, 0.3)
sample['material'].interface.range(1.0, 45.0)
sample['SEI'].thickness.range(10.0, 800.0)
sample['SEI'].material.rho.range(-1.0, 6.0)
sample['SEI'].material.irho.range(0.0, 0.3)
sample['SEI'].interface.range(1.0, 100.0)



probe.intensity=Parameter(value=1.08,name='normalization')
probe.background=Parameter(value=0.0,name='background')
sample['THF'].interface.range(1.0, 100.0)

################################################################################

expt = Experiment(probe=probe, sample=sample)
problem = FitProblem(expt)
