import os
import json
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines

from matplotlib.path import Path
from matplotlib.patches import PathPatch

try:
    from bumps import dream
    import fit_uncertainties
    HAS_BUMPS  = True
except:
    print("No bumps")
    HAS_BUMPS = False


def read_model(model_path, dq=0.027):
    with open(model_path) as fd:
        print(model_path)
        code = fd.read()
        # The refl1d model we are reading was auto-generated. We need to replace
        # the data part in case the data file is no longer available.
        code.replace('reduced_file = "/SNS/users/m2d/__data.txt"', 'Q = np.arange(0.008, 1., 0.01)')
        code.replace('Q, R, dR, dQ = numpy.loadtxt(reduced_file).T', 'dQ = Q*%s' % dq)
        code.replace(', data=(R[i_min:i_max], dR[i_min:i_max])', '')

        exec(code, globals())
        print("Read in %s" % model_path)


def plot_sld(run, title, fit_dir=None, show_cl=True, dq=0.027):
    """
        :param ar_dir: Automated-reduction directory
    """
    sld_file = os.path.join(fit_dir, str(run), "__model-profile.dat")
    if not os.path.isfile(sld_file):
        return
    pre_sld = np.loadtxt(sld_file).T

    if show_cl and HAS_BUMPS:
        # Load the model that was used for fitting
        read_model(os.path.join(fit_dir, str(run), '__model.py'))
        model_path = os.path.join(fit_dir, str(run), '__model')
        state = dream.state.load_state(model_path)
        z_max = pre_sld[0][-1]-pre_sld[0][0]
        print("Z offset = %g;    Z_max = %g" % (pre_sld[0][0], z_max))
        acc_data = fit_uncertainties.load_bumps(model_path, problem, state=state,
                                                trim=10,
                                                z_min=0, #_data[0][0],
                                                z_max=z_max)[0]

        _z, _q = acc_data.quantiles(90)
        plt.fill_between(_z, _q[0][0], _q[0][1], alpha=0.2)
        #z, sld, dsld = acc_data.mean()
        #plt.fill_between(z[1:], sld[1:]-dsld[1:], sld[1:]+dsld[1:],
        #                alpha=0.2)
    plt.plot(pre_sld[0][-1]-pre_sld[0], pre_sld[1], markersize=4, label=title, linewidth=2, )


def plot_fit(run, title, fit_dir=None, ar_dir=None, scale=1):
    data_file = os.path.join(ar_dir, 'REFL_%s_combined_data_auto.txt' % run)
    _data = np.loadtxt(data_file).T

    fit_file = os.path.join(fit_dir, str(run), "__model-refl.dat")
    _fit = np.loadtxt(fit_file).T

    plt.errorbar(_data[0], _data[1]*scale, yerr=_data[2]*scale, linewidth=1, 
                 markersize=4, marker='.', linestyle='', label=title)

    plt.plot(_fit[0], _fit[4]*scale, linewidth=1, markersize=2, marker='', color='grey')


def plot_dyn_data(dynamic_run, initial_state, final_state, 
                  fit_dir=None, ar_dir=None, dyn_data_dir=None, dyn_fit_dir=None, scale=1):
    # Reduced data
    pre_data_file = os.path.join(ar_dir, 'REFL_%s_combined_data_auto.txt' % initial_state)
    pre_data = np.loadtxt(pre_data_file).T

    post_data_file = os.path.join(ar_dir, 'REFL_%s_combined_data_auto.txt' % final_state)
    post_data = np.loadtxt(post_data_file).T

    # Fit results
    pre_fit_file = os.path.join(fit_dir, str(initial_state), "__model-refl.dat")
    pre_fit = np.loadtxt(pre_fit_file).T

    post_fit_file = os.path.join(fit_dir, str(final_state), "__model-refl.dat")
    post_fit = np.loadtxt(post_fit_file).T

    # Dynamic data
    _file_list = sorted(os.listdir(dyn_data_dir))
    fig, ax = plt.subplots(dpi=150, figsize=(5,8))
    plt.subplots_adjust(left=0.15, right=.95, top=0.98, bottom=0.1)

    #plt.plot(pre_data[0], pre_data[1], linewidth=1, color='darkgreen', label='initial')
    plt.errorbar(pre_data[0], pre_data[1], yerr=pre_data[2], linewidth=1, 
                 markersize=2, marker='.', linestyle='',
                 color='darkgreen', label='Pre cycle 1')

    plt.plot(pre_fit[0], pre_fit[4], linewidth=1, markersize=2, marker='', color='black')

    # Get only the files for the run we're interested in
    _good_files = [_f for _f in _file_list if _f.startswith('r%d_t' % dynamic_run)]

    print(len(_good_files))

    scale = 1.
    multiplier = 10
    for _file in _good_files[10:18]:
        if _file.startswith('r%d_t' % dynamic_run):
            scale *= 1
            _data = np.loadtxt(os.path.join(dyn_data_dir, _file)).T
            _data_name, _ = os.path.splitext(_file)
            _time = int(_data_name.replace('r%d_t' % dynamic_run, ''))
            _label = '%d < t < %d s' % (_time, _time+15)

            
            # Get fit if it exists
            fit_file = os.path.join(dyn_fit_dir, _data_name, '__model-refl.dat')

            if os.path.isfile(fit_file):
                fit_data = np.loadtxt(fit_file).T
                plt.plot(fit_data[0], fit_data[4]*scale, markersize=2, marker='', linewidth=1, color='black')

            idx = _data[2]<_data[1]
            plt.errorbar(_data[0][idx], _data[1][idx]*scale,
                         yerr=_data[2][idx]*scale, linewidth=1,
                         markersize=2, marker='.',  linestyle='', label=_label)

            scale *= multiplier

    final_scale = scale/multiplier
    plt.plot(post_fit[0], post_fit[4]*final_scale, linewidth=1, color='darkblue', label='final')
    plt.errorbar(post_data[0], post_data[1]*final_scale, yerr=post_data[2]*final_scale, linewidth=1, 
                 markersize=2, marker='.', linestyle='',
                 color='darkgreen', label='Post cycle 1')

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], frameon=False, prop={'size': 8})
    plt.xlabel('Q ($1/\AA$)', fontsize=15)
    plt.ylabel('Reflectivity', fontsize=15)
    plt.yscale('log')
    plt.xscale('log')
    ax.yaxis.labelpad = 1

    plt.show()


def plot_dyn_sld(file_list, initial_state, final_state, 
                 fit_dir=None, dyn_data_dir=None, dyn_fit_dir=None, model_name='__model',
                 model_file=None):

    fig, ax = plt.subplots(dpi=150, figsize=(5, 4.1))
    plt.subplots_adjust(left=0.15, right=.95, top=0.95, bottom=0.15)

    # Plot initial state
    if initial_state is not None:
        plot_sld(initial_state, 'Initial state', fit_dir=fit_dir, show_cl=True)  

    for _file in file_list:
        _data = np.loadtxt(os.path.join(dyn_fit_dir, str(_file[2]), '%s-profile.dat' % model_name)).T
        
        if HAS_BUMPS:
            mc_file = os.path.join(dyn_fit_dir, str(_file[2]), '%s-chain.mc' % model_name)
            if os.path.isfile(mc_file):
                if model_file is None:
                    model_file = os.path.join(dyn_fit_dir, str(_file[2]), '%s.py' % model_name)
                read_model(model_file)
                model_path = os.path.join(dyn_fit_dir, str(_file[2]), model_name)
                print("Model: %s" % model_path)
                state = dream.state.load_state(model_path)
                z_max = _data[0][-1]-_data[0][0]
                print("Z offset = %g;    Z_max = %g" % (_data[0][0], z_max))
                acc_data = fit_uncertainties.load_bumps(model_path, problem, state=state,
                                                        trim=1000,
                                                        z_min=0, #_data[0][0],
                                                        z_max=z_max)[0]

                _z, _q = acc_data.quantiles(90)
                plt.fill_between(_z, _q[0][0], _q[0][1], alpha=0.2)
            
        _label = '%d < t < %d s' % (int(_file[0]), int(_file[0])+30)
        plt.plot(_data[0][-1]-_data[0], _data[1], markersize=4, 
                 label=_label, linewidth=2, )

    # Plot final OCP
    if final_state is not None:
        plot_sld(final_state, 'Final state', fit_dir=fit_dir, show_cl=True)           
        
    handles, labels = ax.get_legend_handles_labels()
    #plt.legend(frameon=False, prop={'size': 10})
    plt.legend(handles[::-1], labels[::-1], loc='lower right', frameon=False, fontsize=6)
    plt.xlabel('z ($\AA$)', fontsize=14)
    plt.ylabel('SLD ($10^{-6}/\AA^2$)', fontsize=14)
    plt.show()


def trend_data(file_list, initial_state, final_state, 
                 fit_dir=None, dyn_data_dir=None, dyn_fit_dir=None, model_name='__model',
                 model_file=None):
    """
        sei_thick.append(item['sei thickness'][which])
    sei_dthick.append(item['sei thickness']['std'])
    """
    # Get the varying parameters, which are assumed to be the same for all data sets
    par_file = os.path.join(dyn_fit_dir, str(file_list[0][2]), '%s.par' % model_name)
    trend_data = dict()
    trend_err = dict()
    chi2 = []
    timestamp = []

    with open(par_file, 'r') as fd:
        for line in fd.readlines():
            par = ' '.join(line.split(' ')[0:2])
            trend_data[par] = []
            trend_err[par] = []

    # Go through each file and retrieve the parameters
    # 'which' defines the value to select. It can either be 'mean' of 'best'.
    which = 'best'
    for _file in file_list:
        try:
            with open(os.path.join(dyn_fit_dir, str(_file[2]), '%s.err' % model_name)) as fd:
                for l in fd.readlines():
                    if l.startswith('[chisq'):
                        toks = l.split('(')
                        _chi2 = toks[0].replace('[chisq=', '')
                        chi2.append(float(_chi2))
            
            with open(os.path.join(dyn_fit_dir, str(_file[2]), '%s-err.json' % model_name)) as fd:
                m = json.load(fd)
                for par in trend_data.keys():
                    trend_data[par].append(m[par][which])
                    trend_err[par].append(m[par]['std'])

            timestamp.append(float(_file[0]))
        except:
            print("Could not process %s" % _file)

    # Plot trend data
    n_tot = len(trend_data.keys())
    fig, axs = plt.subplots(n_tot,1, dpi=100, figsize=(9,20), sharex=True)

    n_current = 1
    for i, par in enumerate(trend_data.keys()):
        ax = plt.subplot(n_tot, 1, i+1)
        plt.errorbar(timestamp, trend_data[par], yerr=trend_err[par], label=par)
        plt.xlabel('seconds')
        plt.ylabel(par)


def save_parameters():
    # Write human-readable output

    _header = ['Time (sec)', 'Layer 1 rho', 'Layer 1 thickness', 'Layer 1 interface',
               'Layer 2 rho', 'Layer 2 thickness', 'Layer 2 interface', 'THF interface', 'chi2']

    with open(os.path.join(project_dir, 'results-table-193673.md'), 'w') as fd:

        header = '| ' + '|'.join(_header) + '|\n'
        header += '| ' + '|'.join(len(_header)*['---']) + '|\n'
        fd.write(header)

        for i, _time in enumerate(timestamp):
            entry = '| %g ' % (60.0*timestamp[i])
            entry += '| %4.2f ± %4.2f' % (oxide_rho[i], oxide_drho[i])
            entry += '| %4.1f ± %4.1f' % (oxide_thick[i], oxide_dthick[i])
            entry += '| %4.1f ± %4.1f' % (oxide_sigma[i], oxide_dsigma[i])
            entry += '| %4.2f ± %4.2f' % (sei_rho[i], sei_drho[i])
            entry += '| %4.1f ± %4.1f' % (sei_thick[i], sei_dthick[i])
            entry += '| %4.1f ± %4.1f' % (sei_sigma[i], sei_dsigma[i])
            entry += '| %4.1f ± %4.1f' % (solvent_sigma[i], solvent_dsigma[i])
            entry += '| %g |\n' % chi2[i]
            fd.write(entry)

    # Save trend data
    trend_data = [60.0*np.asarray(timestamp), oxide_rho, oxide_drho, oxide_thick, oxide_dthick, oxide_sigma, oxide_dsigma,
                  sei_rho, sei_drho, sei_thick, sei_dthick, sei_sigma, sei_dsigma, solvent_sigma, solvent_dsigma]
    trend_data = np.asarray(trend_data).T

    np.savetxt(os.path.join(project_dir, "results-table-193673.txt"), trend_data)