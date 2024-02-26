
import numpy
import os
from refl1d.names import *
from math import *
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
warnings.simplefilter('ignore', UserWarning)

# run ref_l/207232 #############################################################
# Maximum Q-value ##############################################################
q_min = 0.0
q_max = 1.0

reduced_file = "/SNS/users/pvaiec/__data9224.txt"

Q, R, dR, dQ = numpy.loadtxt(reduced_file).T
i_min = min([i for i in range(len(Q)) if Q[i]>q_min])
i_max = max([i for i in range(len(Q)) if Q[i]<q_max])+1

# SNS data is FWHM
dQ_std = dQ/2.35
probe9224 = QProbe(Q[i_min:i_max], dQ_std[i_min:i_max], data=(R[i_min:i_max], dR[i_min:i_max]))

# Materials ####################################################################
Si = SLD(name='Si', rho=2.07, irho=0.0)
THF = SLD(name='THF', rho=6.052, irho=0.0)
Ti = SLD(name='Ti', rho=-2.217, irho=0.0)
Cu = SLD(name='Cu', rho=6.455, irho=0.0)
CuOx = SLD(name='CuOx', rho=4.778, irho=0.0)


# Film definition ##############################################################
sample9224 = (  THF(0, 18.49) | CuOx(38.13, 9.926) | Cu(560.6, 9.36) | Ti(52.0, 5.938) | Si )

sample9224['Ti'].thickness.range(10.0, 100.0)
sample9224['Ti'].material.rho.range(-3.0, 0.0)
sample9224['Ti'].interface.range(5.0, 20.0)
sample9224['Cu'].thickness.range(400.0, 700.0)
sample9224['Cu'].material.rho.range(6.0, 7.0)
sample9224['Cu'].interface.range(5.0, 45.0)
sample9224['CuOx'].thickness.range(10.0, 100.0)
sample9224['CuOx'].material.rho.range(1.0, 5.0)
sample9224['CuOx'].interface.range(1.0, 10.0)



probe9224.intensity.range(0.9, 1.1)
probe9224.background=Parameter(value=0.0,name='background')
sample9224['THF'].material.rho.range(6.05, 7.0)
sample9224['THF'].interface.range(1.0, 45.0)

################################################################################

expt9224 = Experiment(probe=probe9224, sample=sample9224)

# run ref_l/207239 #############################################################
# Maximum Q-value ##############################################################
q_min = 0.0
q_max = 1.0

reduced_file = "/SNS/users/pvaiec/__data9370.txt"

Q, R, dR, dQ = numpy.loadtxt(reduced_file).T
i_min = min([i for i in range(len(Q)) if Q[i]>q_min])
i_max = max([i for i in range(len(Q)) if Q[i]<q_max])+1

# SNS data is FWHM
dQ_std = dQ/2.35
probe9370 = QProbe(Q[i_min:i_max], dQ_std[i_min:i_max], data=(R[i_min:i_max], dR[i_min:i_max]))

# Materials ####################################################################
Si = SLD(name='Si', rho=2.07, irho=0.0)
THF = SLD(name='THF', rho=6.063, irho=0.0)
Ti = SLD(name='Ti', rho=-2.279, irho=0.0)
Cu = SLD(name='Cu', rho=6.523, irho=0.0)
CuOx = SLD(name='CuOx', rho=4.702, irho=0.0)


# Film definition ##############################################################
sample9370 = (  THF(0, 9.217) | CuOx(40.57, 17.89) | Cu(559.8, 9.606) | Ti(50.11, 3.002) | Si )

sample9370['Ti'].thickness.range(20.0, 55.0)
sample9370['Ti'].material.rho.range(-1.0, -4.0)
sample9370['Ti'].interface.range(3.0, 30.0)
sample9370['Cu'].thickness.range(400.0, 600.0)
sample9370['Cu'].material.rho.range(6.0, 7.0)
sample9370['Cu'].interface.range(1.0, 100.0)
sample9370['CuOx'].thickness.range(5.0, 100.0)
sample9370['CuOx'].material.rho.range(1.0, 6.0)
sample9370['CuOx'].interface.range(1.0, 18.0)



probe9370.intensity.range(0.9, 1.1)
probe9370.background.range(0.0, 1e-06)
sample9370['THF'].material.rho.range(0.0, 6.5)
sample9370['THF'].interface.range(1.0, 100.0)

################################################################################

expt9370 = Experiment(probe=probe9370, sample=sample9370)

# run ref_l/207246 #############################################################
# Maximum Q-value ##############################################################
q_min = 0.0
q_max = 1.0

reduced_file = "/SNS/users/pvaiec/__data9221.txt"

Q, R, dR, dQ = numpy.loadtxt(reduced_file).T
i_min = min([i for i in range(len(Q)) if Q[i]>q_min])
i_max = max([i for i in range(len(Q)) if Q[i]<q_max])+1

# SNS data is FWHM
dQ_std = dQ/2.35
probe9221 = QProbe(Q[i_min:i_max], dQ_std[i_min:i_max], data=(R[i_min:i_max], dR[i_min:i_max]))

# Materials ####################################################################
Si = SLD(name='Si', rho=2.07, irho=0.0)
THF = SLD(name='THF', rho=6.01, irho=0.0)
Ti = SLD(name='Ti', rho=-2.549, irho=0.0)
Cu = SLD(name='Cu', rho=6.494, irho=0.0)
Plated = SLD(name='Plated', rho=2.383, irho=0.07398)
SEI = SLD(name='SEI', rho=3.3, irho=0.2858)


# Film definition ##############################################################
sample9221 = (  THF(0, 75.55) | SEI(340.9, 8.875) | Plated(37.34, 15.27) | Cu(571.3, 9.124) | Ti(52.53, 5.545) | Si )

sample9221['Ti'].thickness.range(10.0, 100.0)
sample9221['Ti'].material.rho.range(-5.0, 1.0)
sample9221['Ti'].interface.range(1.0, 15.0)
sample9221['Cu'].thickness.range(400.0, 700.0)
sample9221['Cu'].material.rho.range(6.0, 7.0)
sample9221['Cu'].interface.range(1.0, 45.0)
sample9221['Plated'].thickness.range(10.0, 100.0)
sample9221['Plated'].material.rho.range(-1.0, 7.0)
sample9221['Plated'].material.irho.range(0.0, 0.3)
sample9221['Plated'].interface.range(1.0, 45.0)
sample9221['SEI'].thickness.range(10.0, 500.0)
sample9221['SEI'].material.rho.range(-1.0, 7.0)
sample9221['SEI'].material.irho.range(0.0, 0.3)
sample9221['SEI'].interface.range(5.0, 45.0)



probe9221.intensity.range(0.9, 1.1)
probe9221.background=Parameter(value=0.0,name='background')
sample9221['THF'].material.rho.range(5.0, 7.0)
sample9221['THF'].interface.range(1.0, 200.0)

################################################################################

expt9221 = Experiment(probe=probe9221, sample=sample9221)

# Constraints ##################################################################
sample9221['THF'].material.rho = sample9224['THF'].material.rho
sample9370['Cu'].material.rho = sample9224['Cu'].material.rho
sample9370['THF'].material.rho = sample9224['THF'].material.rho
sample9221['Cu'].material.rho = sample9224['Cu'].material.rho
sample9370['Ti'].material.rho = sample9224['Ti'].material.rho
sample9370['CuOx'].material.rho = sample9224['CuOx'].material.rho


problem = FitProblem([expt9224,expt9370,expt9221])
