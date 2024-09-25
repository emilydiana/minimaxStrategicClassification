# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import numpy as np
import scipy
import warnings
from src.plotting import save_plots_to_os
from src.hull_to_pareto import determine_pareto_curve


def do_pareto_plot(gammas, max_grp_errs, pop_errs,
                   error_type, pop_error_type,
                   save_plots, dirname,
                   model_type,
                   use_input_commands,
                   tau_list, tau_group_values,
                   data_name='', show_basic_plots=False,
                   val_max_grp_errs=None, val_pop_errs=None, val_trajectories=None, val_bonus_plot_list=None,
                   test_size=0.0):
    """
    Utility function used in main_driver to create a multi-trajectory plot over runs with a range of gamma values,
    and traces the pareto curve of errors resulting mixture models.

    Use argument `show_basic_plots` to enable scatter plots for pairwise relationships between population error,
    max groups error, and gamma, of the final mixture models.
    """
    figures = []
    plt.ion()
    figure_names = []

    # Setup strings for graph titles
    dataset_string = f' on {data_name[0].upper() + data_name[1:]}' if data_name != '' else ''
    learner_types = ['Non-Strategic', '' , 'Na\u00EFve Strategic', '' , '' , 'Ours']
    trials = len(max_grp_errs)
    for lam, tau in enumerate(tau_list):
        figure_names.append('PopError_vs_MaxGroupError_Budget_' + str(tau))
        paretos = []
        paretos_upper = []
        paretos_lower = []
        loc_pop_errs = []
        loc_max_grp_errs = []
        for learner in [0, 2, 5]:
        # Get the pareto curve
            loc_pop_errs.append([])
            loc_max_grp_errs.append([])
            paretos.append([])
            paretos_upper.append([])
            paretos_lower.append([])
            for t in range(trials):
                loc_pop_errs[-1].append(pop_errs[t][tau][learner])
                loc_max_grp_errs[-1].append(max_grp_errs[t][tau][learner])
            loc_av_pop_errs = np.mean(loc_pop_errs[-1],axis=0)
            loc_pop_conf = 1.96*np.std(loc_pop_errs[-1],axis=0)/np.sqrt(trials)
            loc_pop_upper = loc_av_pop_errs + loc_pop_conf
            loc_pop_lower = loc_av_pop_errs - loc_pop_conf
 
            loc_av_max_grp_errs = np.mean(loc_max_grp_errs[-1],axis=0)
            loc_max_conf = 1.96*np.std(loc_max_grp_errs[-1],axis=0)/np.sqrt(trials)
            loc_max_upper = loc_av_max_grp_errs + loc_max_conf
            loc_max_lower = loc_av_max_grp_errs - loc_max_conf
 
            paretos[-1].append(get_pareto(loc_av_pop_errs, loc_av_max_grp_errs))
            paretos_upper[-1].append(get_pareto(loc_pop_upper,loc_max_upper))
            paretos_lower[-1].append(get_pareto(loc_pop_lower, loc_max_lower))
        # Set pop_error string
        pop_error_string = pop_error_type
        if pop_error_type == 'Total':
            pop_error_string = f'0/1 Loss'

        if use_input_commands:
            input('Press `Enter` to display first plot...')
        figures.append(plt.figure())
        plt.title(f'Tau {tau} Pop Error vs. Max Group Error{dataset_string} \n {model_type} with group ratios {tau_group_values}')
        plt.xlabel(f'Pop Error ({pop_error_string})')
        plt.ylabel(f'Max Group Error ({error_type})')
        # Compute and plot pareto curve
        for num, learner in enumerate([0,2,5]):
            plt.scatter(loc_pop_errs[num],loc_max_grp_errs[num], label=learner_types[learner])
            if paretos[num] is not None:
                if paretos[num][0] is not None:
                    plt.plot(paretos[num][0][:, 0], paretos[num][0][:, 1], 'r--', lw=2, label='Pareto Curve: ' + learner_types[learner], alpha=0.5)
                    plt.fill_between(paretos[num][0][:,0],paretos_upper[num][0][:,1],paretos_lower[num][0][:,1],alpha=0.1)
        plt.legend(loc='upper right')
        plt.show()
    
    for lam, tau in enumerate(tau_list):
        figure_names.append('Val_PopError_vs_MaxGroupError_Budget_' + str(tau))
        val_paretos = []
        val_paretos_upper = []
        val_paretos_lower = []
        val_loc_pop_errs = []
        val_loc_max_grp_errs = []
        for learner in [0, 2, 5]:
        # Get the pareto curve
            val_loc_pop_errs.append([])
            val_loc_max_grp_errs.append([])
            val_paretos.append([])
            val_paretos_upper.append([])
            val_paretos_lower.append([])
            for t in range(trials):
                val_loc_pop_errs[-1].append(val_pop_errs[t][tau][learner])
                val_loc_max_grp_errs[-1].append(val_max_grp_errs[t][tau][learner])
            val_loc_av_pop_errs = np.mean(val_loc_pop_errs[-1],axis=0)
            val_loc_pop_conf = 1.96*np.std(val_loc_pop_errs[-1],axis=0)/np.sqrt(trials)
            val_loc_pop_upper = val_loc_av_pop_errs + val_loc_pop_conf
            val_loc_pop_lower = val_loc_av_pop_errs - val_loc_pop_conf
 
            val_loc_av_max_grp_errs = np.mean(val_loc_max_grp_errs[-1],axis=0)
            val_loc_max_conf = 1.96*np.std(val_loc_max_grp_errs[-1],axis=0)/np.sqrt(trials)
            val_loc_max_upper = val_loc_av_max_grp_errs + val_loc_max_conf
            val_loc_max_lower = val_loc_av_max_grp_errs - val_loc_max_conf
            val_paretos[-1].append(get_pareto(val_loc_av_pop_errs, val_loc_av_max_grp_errs[-1]))
            val_paretos_upper[-1].append(get_pareto(val_loc_pop_upper, val_loc_max_upper))
            val_paretos_lower[-1].append(get_pareto(val_loc_pop_lower, val_loc_max_lower))
   
        if use_input_commands:
            input('Press `Enter` to display first plot...')
        figures.append(plt.figure())
        plt.title(f'Validation Tau {tau} Pop Error vs. Max Group Error{dataset_string} \n {model_type} with group ratios {tau_group_values}')
        plt.xlabel(f'Pop Error ({pop_error_string})')
        plt.ylabel(f'Max Group Error ({error_type})')
        # Compute and plot pareto curve
        for num, learner in enumerate([0,2,5]):
            plt.scatter(val_loc_pop_errs[num],val_loc_max_grp_errs[num], label=learner_types[learner])
            if val_paretos[num] is not None and val_paretos[num][0] is not None:
                plt.plot(val_paretos[num][0][:, 0], val_paretos[num][0][:, 1], 'r--', lw=2, label='Pareto Curve: ' + learner_types[learner], alpha=0.5)
                plt.fill_between(val_paretos[num][0][:,0],val_paretos_upper[num][0][:,1],val_paretos_lower[num][0][:,1],alpha=0.1)
        plt.legend(loc='upper right')
        plt.show()
    if save_plots:
        save_plots_to_os(figures, figure_names, dirname, True)
        plt.close('all')


def get_pareto(x, y):
    points = np.zeros((len(x), 2))
    points[:, 0] = x
    points[:, 1] = y
    # Handle the exception and don't print the curve if not necessary
    if (len(x) > 2) and (len(np.unique(x)) > 1) and (len(np.unique(y)) > 1):
        try:
            hull = scipy.spatial.ConvexHull(points)
            pareto = determine_pareto_curve(points[hull.vertices])
            return pareto
        except scipy.spatial.qhull.QhullError:
            warnings.warn('\n WARNING: Scipy exception in qhull. This frequntly happens at high gamma values.'
                          ' Ignoring and continuing... \n ')


def plot_trajectories_from_bonus_plot_data(bonus_plot_list, gammas, model_type, error_type, numsteps,
                                           total_steps_per_gamma,
                                           use_input_commands, test_size=0.0, data_name=''):
    """
    :param bonus_plot_list: List (over gammas) of lists of tuples corresponding to the bonus plots for each run
    :param gammas: list of gammas corresponding to each round
    :return figures, names: list of figures and their names
    """
    figures = []
    names = []

    # Set the first letter to capital if it isn't
    dataset_string = f' on {data_name[0].upper() + data_name[1:]}' if data_name != '' else ''

    try:
        num_bonus_plots = len(bonus_plot_list[0])  # Number of 4-tuples (bonus plots) per value of gamma
    except:
        print(bonus_plot_list)
        warnings.warn('WARNING: Could not index into bonus plots. Skipping and continuing...')
        num_bonus_plots = 0

    # Iterate over the number of bonus plots per individual run
    for plot_index in range(num_bonus_plots):

        if use_input_commands:
            input('Next bonus plot')

        figures.append(plt.figure())  # One figure for 'type' of multi trajectory plot

        # Keep ararys to track the endpoints of the trajectories and eventually plot pareto curve
        endpoints_x = []
        endpoints_y = []

        # Determine values for the name, title, and axes of the multi-trajectory plot
        err_type, _, _, pop_err_type = bonus_plot_list[0][plot_index]
        names.append(f'Multi_Trajectory_Bonus_Plot_for_'
                     f'{err_type if err_type != "0/1 Loss" else "0-1 Loss"}_Group_Error')

        loss_string = ''
        if error_type in ['FP', 'FN']:
            loss_string = f'{error_type} Loss'
        elif error_type.endswith('Log-Loss'):
            loss_string = error_type
        elif error_type == 'Total':
            loss_string = f'0/1 Loss'

        # Rename 'total' error to 0/1 Loss for plotting
        err_string = err_type
        if err_type == 'Total':
            err_string = f'0/1 Loss'

        pop_err_string = pop_err_type
        if pop_err_type == 'Total':
            pop_err_string = f'0/1 Loss'

        validation_string = '' if test_size == 0.0 else f'(Validation: {test_size})'

        plt.title(f'Trajectories over {numsteps} Rounds{dataset_string}' + validation_string +
                  f'\n {model_type} weighted on ' + loss_string)
        plt.xlabel(f'Pop Error ({pop_err_string})')
        plt.ylabel(f'Max Group Error ({err_string})')

        # Plot the trajectories for the 'plot_index'-th error type over all gammas
        for single_run_bonus_plot_tuples, gamma, total_steps in zip(bonus_plot_list, gammas, total_steps_per_gamma):
            err_type, grp_errs, pop_errs, pop_err_type = single_run_bonus_plot_tuples[plot_index]
            x = pop_errs
            y = np.max(grp_errs, axis=1)
            plt.scatter(x, y, c=np.arange(1, total_steps), s=2)  # Plot the individual trajectory
            plt.scatter(x[0], y[0], c='m', s=20)  # Add magenta starting point
            plt.annotate(f'gamma={gamma:.5f}', xy=(x[-1], y[-1]))
            # Add the endpoints for the pareto curve
            endpoints_x.append(x[-1])
            endpoints_y.append(y[-1])

        # Compute and plot pareto curve
        pareto = get_pareto(endpoints_x, endpoints_y)
        if pareto is not None:
            plt.plot(pareto[:, 0], pareto[:, 1], 'r--', lw=2, label='Pareto Curve', alpha=0.5)
        plt.show()

    return figures, names
