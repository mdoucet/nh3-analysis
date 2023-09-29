
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

#reduced_file = "/SNS/users/m2d/__data.txt"

#Q, R, dR, dQ = numpy.loadtxt(reduced_file).T
Q = np.arange(0.008, 0.1, 0.001)
dQ = 0.027*Q
i_min = min([i for i in range(len(Q)) if Q[i]>q_min])
i_max = max([i for i in range(len(Q)) if Q[i]<q_max])+1

# SNS data is FWHM
dQ_std = dQ/2.35
probe = QProbe(Q[i_min:i_max], dQ_std[i_min:i_max])

# Materials ####################################################################
Si = SLD(name='Si', rho=2.07, irho=0.0)
THF = SLD(name='THF', rho=6.187, irho=0.0)
SiOx = SLD(name='SiOx', rho=2.653, irho=0.0)
Ti = SLD(name='Ti', rho=-1.981, irho=0.0)
Cu = SLD(name='Cu', rho=6.471, irho=0.0)
material = SLD(name='material', rho=3.362, irho=0.033)
SEI = SLD(name='SEI', rho=4.354, irho=0.27)


# Film definition ##############################################################
sample = (  THF(0, 59.28) | SEI(261.7, 34.99) | material(104.8, 12.35) | Cu(530.0, 6.168) | Ti(52.83, 8.655) | SiOx(25.06, 2.648) | Si )

probe.intensity=Parameter(value=1.08,name='normalization')
probe.background=Parameter(value=0.0,name='background')

################################################################################

expt = Experiment(probe=probe, sample=sample)

if False:
    expt.sample['material'].thickness.range(10.0, 300.0)
    expt.sample['material'].material.rho.range(-1.0, 10.0)
    expt.sample['material'].material.irho.range(0.0, 0.3)
    expt.sample['material'].interface.range(1.0, 45.0)
    expt.sample['SEI'].thickness.range(10.0, 800.0)
    expt.sample['SEI'].material.rho.range(-1.0, 7)
    expt.sample['SEI'].material.irho.range(0.0, 0.3)
    expt.sample['SEI'].interface.range(1.0, 100.0)
    expt.sample['THF'].interface.range(1.0, 100.0)
else:
    expt.sample['material'].thickness.range(10.0, 300.0)
    expt.sample['material'].material.rho.range(-1.0, 10.0)
    #expt.sample['material'].material.irho.range(0.0, 0.3)
    #expt.sample['material'].interface.range(1.0, 22.0)
    expt.sample['material'].interface.range(5, expt.sample['material'].thickness.value/2+5)


    expt.sample['SEI'].thickness.range(10.0, 800.0)
    expt.sample['SEI'].material.rho.range(-2, 7)
    #expt.sample['SEI'].material.irho.range(0.0, 0.3)
    expt.sample['SEI'].interface.range(1.0, 100.0)

    #expt.sample['THF'].interface.range(1.0, 150.0)
    expt.sample['THF'].interface.range(5, expt.sample['SEI'].thickness.value/2+5)

problem = FitProblem(expt)
