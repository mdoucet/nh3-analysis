import argparse

import os

import numpy as np

import csv

import json

# from dynesty import NestedSampler, DynamicNestedSampler
#
# from dynesty import plotting as dyplot
#
# from dynesty import utils as dyfunc

import ultranest
import ultranest.stepsampler
from ultranest.plot import cornerplot

from bumps.cli import load_model, load_best, make_store, store_overwrite_query, config_matplotlib

from multiprocessing import Pool, cpu_count

# import pandas as pd
# os.environ['OMP_NUM_THREADS'] = '7'
# export OMP_NUM_THREADS=7
# from mpi4py import MPI
# mpi_comm = MPI.COMM_WORLD
# mpi_rank = mpi_comm.Get_rank()


class Sampler:

    def __init__(self, problem, store, live_points=50, frac_remain=0.5):

        # if mpi_rank == 0:
        #     # TODO: need to use the --store option as done in bumps to parse the name
        self.save_name = store
        self.problem = problem
        self.problem.model_reset()
        self.live_points = live_points
        self.frac_remain = frac_remain
        # print(f"frac_remain = {frac_remain}")
        # TODO: don't access problem privates

        self.parameters = self.problem._parameters

        # print(type(self.parameters[0].name))

        # fit space dimension is length of parameter vector
        self.param_names = self.get_param_names()
        self.ndim = len(problem.getp())

        self.nested_sampler = ultranest.ReactiveNestedSampler(
            self.param_names, self.logl, self.prior_transform,
            log_dir=self.save_name, resume=True, vectorized=False
            )

        # nsteps = 2 * len(self.param_names)
        self.nested_sampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(nsteps=400, adaptive_nsteps='move-distance')

    def get_param_names(self):

        param_names = []
        for i, v in enumerate(self.parameters):
            param_names.append(v.name)
        return param_names

    def logl(self, x):

        # Note: only using nllf from experiment; parameter_nllf is handled implicitly by prior_transform

        # and constraints_nllf is ignored for now.

        self.problem.setp(x)

        return -self.problem.model_nllf()

    def prior_transform(self, u):

        x = [pk.bounds.put01(uk) for uk, pk in zip(u, self.parameters)]

        return np.array(x, 'd')

    def sample(self, verbose=True):
        print("MPI:", self.nested_sampler.mpi_size, self.nested_sampler.mpi_rank)
        config_matplotlib(backend='agg')
        self.results = self.nested_sampler.run(frac_remain=self.frac_remain, min_ess=10000, min_num_live_points=self.live_points)

        # results = self.nested_sampler.results

        # print(f"results = {self.results}")
        # print(f"results['samples'] type = {type(results['samples'])}")

        # Calculate parameter means.

        # weights = np.exp(results.logwt -  results.logz[-1])

        # mean, _ = dyfunc.mean_and_cov(results.samples, weights)
        mean = self.results['posterior']['mean']
        best = self.results['maximum_likelihood']['point']
        # print(type(mean))

        # self.problem.setp(mean)
        self.problem.setp(best)
        # Save bumps results
        self.save_bumps_pars(self.problem)
        self.nested_sampler.plot()
        # self.problem.save(f'{self.save_name}')
        self.problem.save(self.problem.output_path)
        # config_matplotlib(backend='agg')
        self.problem.plot(figfile=self.problem.output_path, view='fresnel')

        # Save UltraNest results
        
        # self.corner(results)
        self.save_as_csv(self.results)
        # self.save_as_json(self.results)
        # np.savetxt('mean.dat', mean)
        # cornerplot(self.results)

    # def results_df(self):
    #
    #     df = pd.DataFrame(data=self.results['samples'], columns=self.results['paramnames'])
    #     df.describe()
    #
    #     return df.loc['mean']

    # def corner(self, results):
    #
    #     fig, _ = dyplot.cornerplot(results, color='blue', quantiles=None,
    #
    #                                show_titles=True, max_n_ticks=3,
    #
    #                                truths=np.zeros(self.ndim),
    #
    #                                truth_color='black')
    #
    #
    #
    #     # Label corner plot with parameter names.
    #
    #     axes = np.reshape(np.array(fig.get_axes()), (self.ndim, self.ndim))
    #
    #     for i in range(1, self.ndim):
    #
    #         for j in range(self.ndim):
    #
    #             if i == self.ndim-1:
    #
    #                 axes[i,j].set_xlabel(self.parameters[j].name)
    #
    #             if j == 0:
    #
    #                 axes[i,j].set_ylabel(self.parameters[i].name)
    #
    #
    #
    #     axes[self.ndim-1, self.ndim-1].set_xlabel(self.parameters[-1].name)
    #
    #
    #
    #     fig.savefig(f'{self.save_name}_corner_plot.png')
    #
    #     return fig

    def save_as_csv(self, results):

        w = csv.writer(open(f"{self.save_name}_ultranest_results.csv", "w"))
        for key, val in results.items():
            w.writerow([key, val])

    def save_as_json(self, results):

        with open(f"{self.save_name}_ultranest_results.json", "w") as fp:
            json.dump(results, fp, sort_keys=True, indent=4)

    def save_bumps_pars(self, problem):

        pardata = "".join("%s %.15g\n" % (name, value)
                      for name, value in zip(problem.labels(), problem.getp()))
        # TODO: implement below once --store is done

        # open(self.save_name + ".par", 'wt').write(pardata)

        open(problem.output_path + ".par", 'wt').write(pardata)


def main():

    parser = argparse.ArgumentParser(

        description="run bumps model through ultranest",

        formatter_class=argparse.ArgumentDefaultsHelpFormatter,

        )

    # TODO: add view option
    parser.add_argument('-p', '--pars', type=str, default="", help='retrieve starting point from .par file')

    parser.add_argument('--overwrite', type=bool, default=False, help='overwrite existing results')

    parser.add_argument('--resume', type=str, default=None, help='resume from previous stop point - not really required for ultranest yet')

    parser.add_argument('--batch', type=str, default=None, help='batch run - not implemented yet')

    parser.add_argument('-s', '--store', type=str, default="", help='store folder for UltraNest results')

    parser.add_argument('--livepoints', type=int, default=50, help='number of live points used to sample')

    parser.add_argument('--frac_remain', type=float, default=0.5, help='remaining fraction of likelihood to explore')

    parser.add_argument('modelfile', type=str, nargs=1, help='bumps model file')

    parser.add_argument('modelopts', type=str, nargs='*', help='options passed to the model')

    opts = parser.parse_args()



    # print(opts)

    # print(opts.modelfile)

    problem = load_model(opts.modelfile[0], model_options=opts.modelopts)

    if opts.pars:

        load_best(problem, opts.pars)

    # Make the store path in problem
    make_store(problem, opts, exists_handler=store_overwrite_query)

    sampler = Sampler(problem, opts.store, opts.livepoints, opts.frac_remain)

    sampler.sample()


if __name__ == '__main__':

    main()



