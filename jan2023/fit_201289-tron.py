import sys
import os
import numpy as np
import json
import subprocess
import time

import importlib
from tron.bayesian_analysis import model_utils, fitting_loop

# Data analysis directory
project_dir = os.path.expanduser('~/git/nh3-analysis/jan2023/')

# Upper-level data directory for the time-resolved data
data_dir = '/SNS/REF_L/IPTS-29196/shared/dynamic/Cu-B-Expt-6/30s'

# Directory where we store dynamic fit results
dyn_model_dir = os.path.expanduser(os.path.join(project_dir, 'data', 'results', 'refl1d_output'))

# Auto-reduction directory
ar_dir = '/SNS/REF_L/IPTS-29196/shared/autoreduce/'


# Initial data set and model (starting point)
initial_data_file = os.path.join(ar_dir, 'REFL_201282_combined_data_auto.txt')
initial_data = np.loadtxt(initial_data_file).T

final_data_file = os.path.join(ar_dir, 'REFL_201290_combined_data_auto.txt')
final_data = np.loadtxt(initial_data_file).T

initial_err_file = os.path.join(project_dir, 'dyn-fitting', '201282', '__model-err.json')
initial_expt_file = os.path.join(project_dir, 'dyn-fitting', '201282', '__model-expt.json')

final_err_file = os.path.join(dyn_model_dir, 'REFL_201290-err.json')
final_expt_file = os.path.join(dyn_model_dir, 'REFL_201290-expt.json')

dynamic_run = 201289
store_basename = os.path.join(dyn_model_dir, '%s-dyn/results-30s-bck' % dynamic_run)

results_dir = os.path.join(dyn_model_dir, store_basename)

# Create top-level directory for the dynamic fits
if not os.path.exists(os.path.join(dyn_model_dir, f'{dynamic_run}-dyn')):
    os.makedirs(os.path.join(dyn_model_dir, f'{dynamic_run}-dyn'))


loop = fitting_loop.FittingLoop(data_dir, results_dir=results_dir, model_dir=project_dir, model_name='model-loop-201289',
                                initial_err_file=initial_err_file, initial_expt_file=initial_expt_file,
                                final_err_file=final_err_file, final_expt_file=final_expt_file,
                )

print(loop)
loop.print_initial_final()

first = 0
last = 12

_file_list = sorted(os.listdir(data_dir))
_good_files = [_f for _f in _file_list if _f.startswith('r%d_t' % dynamic_run)]
_good_files = _good_files[first:last]

try:
    loop.fit(_good_files, fit_forward=False)
except:
    print(loop.last_output)
