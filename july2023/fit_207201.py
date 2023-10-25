import sys
import os
import numpy as np
import json
import subprocess
import time

sys.path.append(os.path.expanduser('~/git/analysis_playground/bayesian-fitting'))
import importlib
import model_utils
importlib.reload(model_utils)
import fitting_loop
importlib.reload(fitting_loop)

# Data analysis directory
project_dir = os.path.expanduser('~/git/nh3-analysis/july2023/')

# Auto-reduction directory
ar_dir = '/SNS/REF_L/IPTS-30384/shared/autoreduce/'

# Directory where the time-resolved data is found
dyn_data_dir = '/SNS/REF_L/IPTS-30384/shared/dynamic/Cu-G/30s'

# Directory where we store dynamic fit results
dyn_model_dir = os.path.expanduser(os.path.join(project_dir, 'dyn-fitting'))

dynamic_run = 207201 # Cycle 1

# Directory where to store time-resolved fit results
results_dir = os.path.join(dyn_model_dir, '%s-dyn/results-30s-bck' % dynamic_run)



# Initial data set and model (starting point)
initial_data_file = os.path.join(ar_dir, 'REFL_207268_combined_data_auto.txt')
initial_data = np.loadtxt(initial_data_file).T

# Final data set
final_data_file = os.path.join(ar_dir, 'REFL_207282_combined_data_auto.txt')
final_data = np.loadtxt(initial_data_file).T

# Initial model
initial_err_file = os.path.join(dyn_model_dir, '207194', '__model-err.json')
initial_expt_file = os.path.join(dyn_model_dir, '207194', '__model-expt.json')

# Final model
final_err_file = os.path.join(dyn_model_dir, '207202', '__model-err.json')
final_expt_file = os.path.join(dyn_model_dir, '207202', '__model-expt.json')

print("Starting loop")
loop = fitting_loop.FittingLoop(dyn_data_dir, results_dir=results_dir, model_dir=project_dir, model_name='model-loop-207201',
                                initial_err_file=initial_err_file, initial_expt_file=initial_expt_file,
                                final_err_file=final_err_file, final_expt_file=final_expt_file)

try:
    print("----------------------------------------------------------------")
    loop.print_initial_final()
    print("----------------------------------------------------------------")
except:
    print(loop.last_output)
    raise

PROCESS_ALL_DATA = True

first = 0
last = 12

if PROCESS_ALL_DATA:
    _file_list = sorted(os.listdir(dyn_data_dir))

    # Get only the files for the run we're interested in
    _good_files = [_f for _f in _file_list if _f.startswith('r%d_t' % dynamic_run)]
    _good_files = _good_files[first:last]

try:
    loop.fit(_good_files, fit_forward=False)
except:
    print("----------------------------------------------------------------")
    for l in loop.last_output.stderr.split('\n'):
        print('  ', l)
    print("----------------------------------------------------------------")
    raise
