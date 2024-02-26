
import numpy
import os
from refl1d.names import *
from math import *
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
warnings.simplefilter('ignore', UserWarning)

# run ref_l/207012 #############################################################
# Maximum Q-value ##############################################################
q_min = 0.0
q_max = 1.0

reduced_file = "/SNS/users/pvaiec/__data9192.txt"

Q, R, dR, dQ = numpy.loadtxt(reduced_file).T
i_min = min([i for i in range(len(Q)) if Q[i]>q_min])
i_max = max([i for i in range(len(Q)) if Q[i]<q_max])+1

# SNS data is FWHM
dQ_std = dQ/2.35
probe9192 = QProbe(Q[i_min:i_max], dQ_std[i_min:i_max], data=(R[i_min:i_max], dR[i_min:i_max]))

# Materials ####################################################################
Si = SLD(name='Si', rho=2.07, irho=0.0)
THF = SLD(name='THF', rho=6.112, irho=0.0)
Ti = SLD(name='Ti', rho=-1.955, irho=0.0)
Cu = SLD(name='Cu', rho=6.482, irho=0.0)
CuOx = SLD(name='CuOx', rho=4.2, irho=0.0)


# Film definition ##############################################################
sample9192 = (  THF(0, 20.68) | CuOx(46.54, 20.25) | Cu(546.9, 8.139) | Ti(49.82, 9.084) | Si )

sample9192['Ti'].thickness.range(10.0, 100.0)
sample9192['Ti'].material.rho.range(-3.0, 0.0)
sample9192['Ti'].interface.range(5.0, 20.0)
sample9192['Cu'].thickness.range(400.0, 700.0)
sample9192['Cu'].material.rho.range(6.0, 7.0)
sample9192['Cu'].interface.range(5.0, 45.0)
sample9192['CuOx'].thickness.range(5.0, 100.0)
sample9192['CuOx'].material.rho.range(4.2, 7.0)
sample9192['CuOx'].interface.range(1.0, 30.0)



probe9192.intensity=Parameter(value=1.0,name='normalization')
probe9192.background=Parameter(value=0.0,name='background')
sample9192['THF'].material.rho.range(6.0, 7.0)
sample9192['THF'].interface.range(1.0, 45.0)

################################################################################

expt9192 = Experiment(probe=probe9192, sample=sample9192)

# run ref_l/207020 #############################################################
# Maximum Q-value ##############################################################
q_min = 0.0
q_max = 1.0

reduced_file = "/SNS/users/pvaiec/__data9194.txt"

Q, R, dR, dQ = numpy.loadtxt(reduced_file).T
i_min = min([i for i in range(len(Q)) if Q[i]>q_min])
i_max = max([i for i in range(len(Q)) if Q[i]<q_max])+1

# SNS data is FWHM
dQ_std = dQ/2.35
probe9194 = QProbe(Q[i_min:i_max], dQ_std[i_min:i_max], data=(R[i_min:i_max], dR[i_min:i_max]))

# Materials ####################################################################
Si = SLD(name='Si', rho=2.07, irho=0.0)
THF = SLD(name='THF', rho=5.505, irho=0.0)
Ti = SLD(name='Ti', rho=-2.444, irho=0.0)
Cu = SLD(name='Cu', rho=6.478, irho=0.0)
Plated = SLD(name='Plated', rho=2.06, irho=0.2999)
SEI = SLD(name='SEI', rho=3.759, irho=0.2945)


# Film definition ##############################################################
sample9194 = (  THF(0, 72.37) | SEI(110.4, 16.35) | Plated(60.22, 18.57) | Cu(546.5, 15.7) | Ti(46.29, 4.057) | Si )

sample9194['Ti'].thickness.range(10.0, 100.0)
sample9194['Ti'].material.rho.range(-3.0, 0.0)
sample9194['Ti'].interface.range(1.0, 15.0)
sample9194['Cu'].thickness.range(400.0, 700.0)
sample9194['Cu'].material.rho.range(6.0, 7.0)
sample9194['Cu'].interface.range(1.0, 45.0)
sample9194['Plated'].thickness.range(10.0, 100.0)
sample9194['Plated'].material.rho.range(-1.0, 7.0)
sample9194['Plated'].material.irho.range(0.0, 0.3)
sample9194['Plated'].interface.range(1.0, 45.0)
sample9194['SEI'].thickness.range(10.0, 200.0)
sample9194['SEI'].material.rho.range(-1.0, 7.0)
sample9194['SEI'].material.irho.range(0.0, 0.3)
sample9194['SEI'].interface.range(1.0, 45.0)



probe9194.intensity.range(0.9, 1.1)
probe9194.background.range(0.0, 1e-06)
sample9194['THF'].material.rho.range(5.0, 7.0)
sample9194['THF'].interface.range(1.0, 100.0)

################################################################################

expt9194 = Experiment(probe=probe9194, sample=sample9194)

# Constraints ##################################################################
sample9194['THF'].material.rho = sample9192['THF'].material.rho
sample9194['Cu'].material.rho = sample9192['Cu'].material.rho


problem = FitProblem([expt9192,expt9194])
