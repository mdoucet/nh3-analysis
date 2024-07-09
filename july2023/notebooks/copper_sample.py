import numpy as np
import json

import matplotlib as mpl
from matplotlib import pyplot as plt

from refl1d.names import *
from refl1d import errors

from bumps.fitters import fit
from bumps.dream.stats import var_stats
from bumps.dream import entropy


class Measurement:
    def __init__(self, data_file, n_sample=1000):
        self.data_file = data_file
        self.experiment = None
        self.n_sample = n_sample
        self.thf_sld = 6.09
        self.cu_sld = 6.425
    
    def load_data(self):
        q, r, dr, dq = np.loadtxt(self.data_file).T
        self.q = q
        self.r = r
        self.dr = dr
        self.dq = dq

        dQ_std = dq/2.35
        probe = QProbe(q, dQ_std, data=(r, dr))
        return probe

    def get_sample(self):
        # Materials ####################################################################
        Si = SLD(name='Si', rho=2.07, irho=0.0)
        THF = SLD(name='THF', rho=self.thf_sld, irho=0.0)
        siox = SLD(name='siox', rho=3.2, irho=0.0)
        Ti = SLD(name='Ti', rho=-2.0, irho=0.0)
        Cu = SLD(name='Cu', rho=self.cu_sld, irho=0.0)
        material = SLD(name='material', rho=2, irho=0.0)
        SEI = SLD(name='SEI', rho=3., irho=0.0)

        # Film definition ##############################################################
        sample = (  THF(0, 54.16) | SEI(222, 9.74) | material(42, 9.74) | Cu(560.6, 15.24) | Ti(51, 11.1) | siox(32, 2.92) | Si )
        return sample

    def get_experiment(self):
        probe = self.load_data()
        sample = self.get_sample()
        self.experiment = Experiment(probe=probe, sample=sample)
        return self.experiment

    def set_ranges(self):
        probe = self.experiment.probe
        sample = self.experiment.sample

        sample['siox'].thickness.range(1.0, 35.0)
        sample['siox'].interface.range(1.0, 11.0)
        sample['siox'].material.rho.range(1.3, 4.2)
        sample['Ti'].thickness.range(10.0, 100.0)
        sample['Ti'].interface.range(1.0, 10.0)
        sample['Ti'].material.rho.range(-1.75, -2.5)
        sample['Cu'].thickness.range(10.0, 800.0)
        sample['Cu'].interface.range(5.0, 22.0)
        #sample['Cu'].material.rho.range(6.3, 7.0)
        sample['material'].thickness.range(10.0, 50.0)
        sample['material'].material.rho.range(1.0, 6)
        sample['material'].interface.range(1.0, 33.0)
        sample['SEI'].thickness.range(100.0, 350.0)
        sample['SEI'].material.rho.range(0.0, 7.0)
        #sample['SEI'].material.irho.range(0.0, 0.2)
        sample['SEI'].interface.range(1.0, 25.0)

        probe.intensity.range(0.5, 1.15)
        #probe.background.range(0.0, 1e-05)
        sample['THF'].interface.range(1.0, 120.0)
        #sample['THF'].material.rho.range(5.5, 6.5)

    def fit(self):
        if self.experiment is None:
            self.get_experiment()
        self.set_ranges()
        self.problem = FitProblem(self.experiment)
        self.results = fit(self.problem, method='dream', samples=self.n_sample, burn=5000, alpha=1, verbose=1)
        return self.results
    
    def plot(self, portion=1):
        q, r = self.experiment.reflectivity()

        fig, axs = plt.subplots(2,1, dpi=100, figsize=(6,8), sharex=False)
        ax = plt.subplot(2, 1, 1)
        plt.errorbar(self.q, self.r, yerr=self.dr, label='data', linestyle='', marker='.')
        plt.plot(q, r, color='gray', label='fit', zorder=10)

        plt.gca().legend()
        plt.xlabel('q [$1/\AA$]')
        plt.ylabel('R(q)')
        plt.xscale('log')
        plt.yscale('log')

        z, best, low, high = get_sld_contour(self.problem, self.results.state,
                                             cl=90, portion=portion, align=-1)[0]

        ax = plt.subplot(2, 1, 2)
        _z = z[-1]-z+0
        plt.plot(_z, best, markersize=4, label='best', linewidth=2,)
        plt.fill_between(_z, low, high, alpha=0.2, color=plt.gca().lines[-1].get_color())

        # Store plotting data
        self.z = _z
        self.sld = best
        self.sld_low = low
        self.sld_high = high

        self.r_calc = r
    
def write_markdown_table(measurement, output_file):
    draw = measurement.results.state.draw(portion=1)
    all_vstats = var_stats(draw)
    S_kde = entropy.kde_entropy_sklearn_gmm(draw.points, n_est=1000)

    fit_params = {}
    for v in all_vstats:
        fit_params[v.label] = {'best': v.best, 'std': v.std}

    fit_data_file = output_file.replace('.md', '.json')
    with open(fit_data_file, 'w') as f:
        fit_data = dict(z=list(measurement.z),
                        sld=list(measurement.sld),
                        sld_low=list(measurement.sld_low),
                        sld_high=list(measurement.sld_high),
                        q=list(measurement.q),
                        r_calc=list(measurement.r_calc))
        json.dump(fit_data, f)

    with open(output_file, 'w') as output:
        # Write header
        entry = measurement.data_file + '\n\n'
        chi2 = measurement.problem.chisq()
        print("Chi2: ", chi2, "S: ", S_kde)
        if measurement.experiment.probe.intensity.fixed:
            entry += 'Intensity: %4.2f\n\n' % measurement.experiment.probe.intensity.value
        else:
            entry += 'Intensity: %4.2f ± %4.2f\n\n' % (fit_params[measurement.experiment.probe.intensity.name]['best'],
                                                    fit_params[measurement.experiment.probe.intensity.name]['std'])
        entry += '$\chi^2$: %4.2g\n\n' % chi2
        entry += 'Entropy: %4.3g\n\n' % S_kde
        output.write(entry)
        
        headers = ['Layer', 'Thickness (Å)',
                    'SLD ($Å^{-2}$)', 'iSLD ($Å^{-2}$)', 
                    'Roughness (Å)']
        header = '| ' + ' | '.join(headers) + ' |\n'
        header += '| ' + ' | '.join((len(headers))*['---']) + ' |\n'
        output.write(header)

        for l in measurement.experiment.sample:

            entry = '| %20s ' % l.name
            if l.thickness.fixed:
                entry += '| %4.2f ' % l.thickness.value
            else:
                entry += '| %4.2f ± %4.2f ' % (fit_params[l.thickness.name]['best'],
                                                fit_params[l.thickness.name]['std'])

            if l.material.rho.fixed:
                entry += '| %4.2f ' % l.material.rho.value
            else:
                entry += '| %4.2f ± %4.2f ' % (fit_params[l.material.rho.name]['best'],
                                                fit_params[l.material.rho.name]['std'])

            if l.material.irho.fixed:
                entry += '| %4.2f ' % l.material.irho.value
            else:
                entry += '| %4.2f ± %4.2f ' % (fit_params[l.material.irho.name]['best'],
                                                fit_params[l.material.irho.name]['std'])

            if l.interface.fixed:
                entry += '| %4.2f |\n' % l.interface.value
            else:
                entry += '| %4.2f ± %4.2f |\n' % (fit_params[l.interface.name]['best'],
                                                fit_params[l.interface.name]['std'])


            output.write(entry)


class Measurement_K_OCV3(Measurement):
    def get_sample(self):
        # Materials ####################################################################
        Si = SLD(name='Si', rho=2.07, irho=0.0)
        THF = SLD(name='THF', rho=self.thf_sld, irho=0.0)
        siox = SLD(name='siox', rho=3.2, irho=0.0)
        Ti = SLD(name='Ti', rho=-2.0, irho=0.0)
        Cu = SLD(name='Cu', rho=self.cu_sld, irho=0.0)
        material = SLD(name='material', rho=2, irho=0.0)
        SEI = SLD(name='SEI', rho=3., irho=0.0)
    
        # Film definition ##############################################################
        sample = (  THF(0, 54.16) | SEI(222, 9.74) | material(42, 9.74) | Cu(560.6, 15.24) | Ti(51, 11.1) | siox(32, 2.92) | Si )
        return sample
    
    def set_ranges(self):
        probe = self.experiment.probe
        sample = self.experiment.sample

        sample['siox'].thickness.range(1.0, 35.0)
        sample['siox'].interface.range(1.0, 51.0)
        sample['siox'].material.rho.range(-2, 2.5)

        sample['Ti'].thickness.range(10.0, 100.0)
        sample['Ti'].interface.range(1.0, 20.0)
        sample['Ti'].material.rho.range(-2, -4.5)
        sample['Cu'].thickness.range(10.0, 800.0)
        sample['Cu'].interface.range(5.0, 22.0)

        sample['material'].thickness.range(10.0, 120.0)
        sample['material'].material.rho.range(0.0, 6)
        sample['material'].interface.range(1.0, 33.0)
        sample['SEI'].thickness.range(100.0, 350.0)
        sample['SEI'].material.rho.range(0.0, 7.0)

        sample['SEI'].interface.range(1.0, 25.0)

        probe.intensity.range(0.5, 1.15)

        sample['THF'].interface.range(1.0, 120.0)

class Measurement_K_OCV4(Measurement):
    def get_sample(self):
        # Materials ####################################################################
        Si = SLD(name='Si', rho=2.07, irho=0.0)
        THF = SLD(name='THF', rho=self.thf_sld, irho=0.0)
        siox = SLD(name='siox', rho=3.2, irho=0.0)
        Ti = SLD(name='Ti', rho=-2.0, irho=0.0)
        Cu = SLD(name='Cu', rho=self.cu_sld, irho=0.0)
        material = SLD(name='material', rho=2, irho=0.0)
        SEI = SLD(name='SEI', rho=3., irho=0.00)
    
        # Film definition ##############################################################
        sample = (  THF(0, 54.16) | SEI(222, 9.74) | material(42, 9.74) | Cu(560.6, 15.24) | Ti(51, 11.1) | siox(32, 2.92) | Si )
        return sample
    

    def set_ranges(self):
        probe = self.experiment.probe
        sample = self.experiment.sample

        sample['siox'].thickness.range(1.0, 55.0)
        sample['siox'].interface.range(1.0, 51.0)
        sample['siox'].material.rho.range(-2, 4.2)

        sample['Ti'].thickness.range(10.0, 80.0)
        sample['Ti'].interface.range(1.0, 20.0)
        sample['Ti'].material.rho.range(-1.75, -3.5)
        sample['Cu'].thickness.range(10.0, 800.0)
        sample['Cu'].interface.range(12.0, 32.0)

        sample['material'].thickness.range(10.0, 100.0)
        sample['material'].material.rho.range(1.0, 6)
        sample['material'].interface.range(1.0, 33.0)
        sample['SEI'].thickness.range(100.0, 350.0)
        sample['SEI'].material.rho.range(0.0, 7.0)

        sample['SEI'].interface.range(1.0, 25.0)

        probe.intensity.range(0.5, 1.15)

        sample['THF'].interface.range(1.0, 120.0)


class Measurement_CuF(Measurement):
    def set_ranges(self):
        probe = self.experiment.probe
        sample = self.experiment.sample

        sample['siox'].thickness.range(10.0, 30.0)
        sample['siox'].interface.range(1.0, 20.0)
        sample['siox'].material.rho.range(1.3, 4.0)
        sample['Ti'].thickness.range(10.0, 100.0)
        sample['Ti'].interface.range(1.0, 10.0)
        sample['Ti'].material.rho.range(-4, 0)
        sample['Cu'].thickness.range(10.0, 800.0)
        sample['Cu'].interface.range(5.0, 22.0)
        #sample['Cu'].material.rho.range(6.3, 7.0)
        sample['material'].thickness.range(10.0, 80.0)
        sample['material'].material.rho.range(1.0, 8.0)
        sample['material'].interface.range(5.0, 33.0)
        sample['SEI'].thickness.range(100.0, 350.0)
        sample['SEI'].material.rho.range(0.0, 7.0)
        #sample['SEI'].material.irho.range(0.0, 0.2)
        sample['SEI'].interface.range(1.0, 25.0)

        probe.intensity.range(0.5, 1.15)
        #probe.background.range(0.0, 1e-05)
        sample['THF'].interface.range(1.0, 220.0)
        #sample['THF'].material.rho.range(5.5, 6.5)

class Measurement_OCV1(Measurement):
    def get_sample(self):
        # Materials ####################################################################
        Si = SLD(name='Si', rho=2.07, irho=0.0)
        THF = SLD(name='THF', rho=6.13, irho=0.0)
        siox = SLD(name='siox', rho=3.2, irho=0.0)
        Ti = SLD(name='Ti', rho=-2.0, irho=0.0)
        Cu = SLD(name='Cu', rho=6.446, irho=0.0)
        material = SLD(name='material', rho=4, irho=0.0)


        # Film definition ##############################################################
        sample = (  THF(0, 54.16) | material(33, 9.74) | Cu(560.6, 15.24) | Ti(51, 11.1) | siox(32, 2.92) | Si )
        return sample

    def set_ranges(self):
        probe = self.experiment.probe
        sample = self.experiment.sample

        sample['siox'].thickness.range(10.0, 45.0)
        sample['siox'].interface.range(1.0, 11.0)
        sample['siox'].material.rho.range(1.3, 4.0)
        sample['Ti'].thickness.range(10.0, 100.0)
        sample['Ti'].interface.range(1.0, 10.0)
        sample['Ti'].material.rho.range(-1, -4)
        sample['Cu'].thickness.range(10.0, 800.0)
        sample['Cu'].interface.range(11.0, 22.0)
        sample['Cu'].material.rho.range(6.3, 7.0)
        sample['material'].thickness.range(10.0, 200.0)
        sample['material'].material.rho.range(1.0, 8.0)
        sample['material'].interface.range(1.0, 33.0)

        probe.intensity.range(0.8, 1.15)
        #probe.background.range(0.0, 1e-05)
        sample['THF'].interface.range(1.0, 20.0)
        sample['THF'].material.rho.range(5.5, 6.5)

class Measurement_CuG_OCV1(Measurement):
    def get_sample(self):
        # Materials ####################################################################
        Si = SLD(name='Si', rho=2.07, irho=0.0)
        THF = SLD(name='THF', rho=6.13, irho=0.0)
        siox = SLD(name='siox', rho=3.2, irho=0.0)
        Ti = SLD(name='Ti', rho=-2.0, irho=0.0)
        Cu = SLD(name='Cu', rho=6.446, irho=0.0)
        material = SLD(name='material', rho=4, irho=0.0)


        # Film definition ##############################################################
        sample = (  THF(0, 54.16) | Cu(560.6, 15.24) | Ti(51, 11.1) | siox(32, 2.92) | Si )
        return sample

    def set_ranges(self):
        probe = self.experiment.probe
        sample = self.experiment.sample

        sample['siox'].thickness.range(10.0, 45.0)
        sample['siox'].interface.range(1.0, 11.0)
        sample['siox'].material.rho.range(-1, 4.0)
        sample['Ti'].thickness.range(10.0, 100.0)
        sample['Ti'].interface.range(1.0, 10.0)
        sample['Ti'].material.rho.range(-1, -4)
        sample['Cu'].thickness.range(400.0, 800.0)
        sample['Cu'].interface.range(11.0, 22.0)
        sample['Cu'].material.rho.range(6.3, 7.0)
        #sample['material'].thickness.range(10.0, 200.0)
        #sample['material'].material.rho.range(1.0, 8.0)
        #sample['material'].interface.range(1.0, 33.0)

        probe.intensity.range(0.8, 1.15)
        #probe.background.range(0.0, 1e-05)
        sample['THF'].interface.range(1.0, 20.0)
        sample['THF'].material.rho.range(-1, 1)


class Measurement_CuG_OCV2(Measurement):
    def get_sample(self):
        # Materials ####################################################################
        Si = SLD(name='Si', rho=2.07, irho=0.0)
        THF = SLD(name='THF', rho=self.thf_sld, irho=0.0)
        siox = SLD(name='siox', rho=3.2, irho=0.0)
        Ti = SLD(name='Ti', rho=-2.0, irho=0.0)
        Cu = SLD(name='Cu', rho=self.cu_sld, irho=0.0)
        SEI = SLD(name='SEI', rho=3., irho=0.0)

        # Film definition ##############################################################
        sample = (  THF(0, 54.16) | SEI(222, 9.74) |Cu(560.6, 15.24) | Ti(51, 11.1) | siox(32, 2.92) | Si )
        return sample

    def set_ranges(self):
        probe = self.experiment.probe
        sample = self.experiment.sample

        sample['siox'].thickness.range(1.0, 25.0)
        sample['siox'].interface.range(1.0, 11.0)
        sample['siox'].material.rho.range(0.3, 4.2)
        sample['Ti'].thickness.range(10.0, 100.0)
        sample['Ti'].interface.range(1.0, 15.0)
        sample['Ti'].material.rho.range(0, -4)
        sample['Cu'].thickness.range(10.0, 800.0)
        sample['Cu'].interface.range(5.0, 22.0)

        sample['SEI'].thickness.range(10.0, 350.0)
        sample['SEI'].material.rho.range(0.0, 7.0)
        sample['SEI'].interface.range(1.0, 25.0)

        probe.intensity.range(0.5, 1.15)
        sample['THF'].interface.range(1.0, 120.0)

class MeasurementHighCurrent(Measurement):
    def get_sample(self):
        # Materials ####################################################################
        Si = SLD(name='Si', rho=2.07, irho=0.0)
        THF = SLD(name='THF', rho=self.thf_sld, irho=0.0)
        siox = SLD(name='siox', rho=3.2, irho=0.0)
        Ti = SLD(name='Ti', rho=-2.0, irho=0.0)
        Cu = SLD(name='Cu', rho=self.cu_sld, irho=0.0)
        SEI = SLD(name='SEI', rho=3., irho=0.0)

        # Film definition ##############################################################
        sample = (  THF(0, 54.16) | SEI(222, 9.74) |Cu(560.6, 15.24) | Ti(51, 11.1) | siox(32, 2.92) | Si )
        return sample

    def set_ranges(self):
        probe = self.experiment.probe
        sample = self.experiment.sample

        sample['siox'].thickness.range(10.0, 45.0)
        sample['siox'].interface.range(1.0, 11.0)
        sample['siox'].material.rho.range(1.3, 4.2)
        sample['Ti'].thickness.range(10.0, 100.0)
        sample['Ti'].interface.range(1.0, 10.0)
        sample['Ti'].material.rho.range(-1.75, -4)
        sample['Cu'].thickness.range(10.0, 800.0)
        sample['Cu'].interface.range(5.0, 22.0)

        sample['SEI'].thickness.range(100.0, 350.0)
        sample['SEI'].material.rho.range(0.0, 7.0)
        sample['SEI'].interface.range(1.0, 25.0)

        probe.intensity.range(0.5, 1.15)
        sample['THF'].interface.range(1.0, 120.0)

# Adapted from bumps
"""
  Show an uncertainty band for an SLD profile.
  This currently works for inverted geometry and fixed substrate roughness, as it aligns
  the profiles to that point before doing the statistics.
"""
def get_sld_contour(problem, state, cl=90, npoints=1000, portion=1, index=1, align='auto'):
    points, _logp = state.sample(portion=portion)
    original = problem.getp()
    _profiles, slabs, Q, residuals = errors.calc_errors(problem, points)
    problem.setp(original)
    
    profiles = errors.align_profiles(_profiles, slabs, align)

    # Group 1 is rho
    # Group 2 is irho
    # Group 3 is rhoM
    contours = []
    for model, group in profiles.items():
        ## Find limits of all profiles
        z = np.hstack([line[0] for line in group])
        zp = np.linspace(np.min(z), np.max(z), npoints)

        # Columns are z, best, low, high
        data, cols = errors._build_profile_matrix(group, index, zp, [cl])
        contours.append(data)
    return contours
