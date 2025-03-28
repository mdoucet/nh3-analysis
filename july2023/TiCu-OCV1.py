import os
import numpy as np
from bumps.fitters import fit
from refl1d.names import *


def create_fit_experiment(q, dq, data, errors, has_SEI=True):
    """
    Create a fit experiment with the given data and parameters.
    Parameters:
        q (array): Q values.
        dq (array): dQ values.
        data (array): Data values.
        errors (array): Error values.
        has_SEI (bool): Flag to include a double-SEI at the surface.
    Returns:
        M (Experiment): The created experiment.
    """
    # Go from FWHM to sigma
    dq /= 2.355

    # The QProbe object represents the beam
    probe = QProbe(q, dq, data=(data, errors))
    probe.intensity = Parameter(value=1.0, name="intensity")
    probe.intensity.pm(0.05)

    THF = SLD("THF", rho=6.33)
    Si = SLD("Si", rho=2.07)
    Ti = SLD("Ti", rho=-2.5)
    SiOx = SLD("SiOx", rho=3.2)
    Cu = SLD("Cu", rho=8)
    material = SLD(name="material", rho=7.8, irho=0.0)
    SEI = SLD(name="SEI", rho=5.6, irho=0.0)

    if has_SEI:
        sample = (
            THF(0, 5) | SEI(30, 20) | material(42, 4) | Cu(500, 10) | Ti(44, 5) | Si
        )
    else:
        sample = THF(0, 5) | material(42, 4) | Cu(500, 10) | Ti(44, 5) | Si

    M = Experiment(sample=sample, probe=probe)

    sample["THF"].material.rho.range(4.5, 6.4)
    sample["THF"].interface.range(1, 25)

    sample["Ti"].thickness.range(30.0, 60.0)
    sample["Ti"].material.rho.range(-3.0, -1)
    sample["Ti"].interface.range(1.0, 22.0)

    sample["material"].thickness.range(10.0, 200.0)
    sample["material"].material.rho.range(5.0, 12)
    sample["material"].interface.range(1.0, 33.0)

    sample["Cu"].thickness.range(1.0, 1000.0)
    sample["Cu"].material.rho.range(2.0, 12)
    sample["Cu"].interface.range(1.0, 12.0)

    if has_SEI:
        sample["SEI"].thickness.range(5.0, 150.0)
        sample["SEI"].material.rho.range(0.0, 12.0)
        sample["SEI"].interface.range(1.0, 25.0)

    return M


data_dir = "/Users/m2d/git/nh3-analysis/july2023/data/reduced/"

data_file = os.path.join(data_dir, "REFL_206907_combined_data_auto.txt")
_refl = np.loadtxt(data_file).T

experiment = create_fit_experiment(
    _refl[0], _refl[3], _refl[1], _refl[2], has_SEI=False
)
problem = FitProblem(experiment)
