
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
THF = SLD(name='THF', rho=6.215, irho=0.0)
SiOx = SLD(name='SiOx', rho=2.87, irho=0.0)
Ti = SLD(name='Ti', rho=-2.999, irho=0.0)
Cu = SLD(name='Cu', rho=6.486, irho=0.0)
material = SLD(name='material', rho=2.854, irho=0.1)
SEI = SLD(name='SEI', rho=3.143, irho=0.1)


# Film definition ##############################################################
sample = (  THF(0, 78.04) | SEI(258.2, 6.24) | material(85.26, 19.53) | Cu(532.5, 6.591) | Ti(52.67, 10.52) | SiOx(23.69, 2.277) | Si )


sample['Ti'].material.rho.range(-3.5, 0)
sample['Cu'].thickness.range(sample['Cu'].thickness.value*0.97, sample['Cu'].thickness.value*1.03)
sample['material'].thickness.range(25.0, 110.0)
sample['material'].material.rho.range(-1.0, 5.5) # Cu-K was 4.0
sample['material'].interface.range(1.0, 25.0)
sample['SEI'].thickness.range(55.0, 300.0)
sample['SEI'].material.rho.range(3, 6.3)
sample['SEI'].interface.range(8.0, 25.0)
sample['THF'].interface.range(15.0, 150.0)


probe.intensity=Parameter(value=1.041,name='normalization')
probe.background=Parameter(value=0.0,name='background')

################################################################################

expt = Experiment(probe=probe, sample=sample)
problem = FitProblem(expt)
