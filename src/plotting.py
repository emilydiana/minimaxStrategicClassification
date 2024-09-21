# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import numpy as np
import os
from src.save_plots import save_plots_to_os
from scipy.spatial.distance import pdist, squareform


def do_plotting(display_plots, save_plots, use_input_commands, numsteps, group_names, group_type,
                show_legend, error_type, data_name, model_string,
                agg_poperrs, agg_grouperrs, groupweights, pop_error_type, bonus_plots,
                dirname, tau, strategic_learner, multi_group=False,
                validation=False, equal_error=False, curr_idx = 0):
    """
    Helper function for minimaxML that creates the relevant plots for a single run of the simulation.
    """

    # Create a list of all figures we want to save for later which will be passed into a function

    figures = []
    str_algs= ['Non-Strategic', '' , 'Na\u00EFve Strategic', '' , '' , 'Ours']
    alg_name = f'{str_algs[curr_idx]}'
    figure_names = [f'{alg_name}_PopError_vs_Rounds', f'{alg_name}_GroupError_vs_Rounds', 
                    f'{alg_name}_GroupWeights_vs_Rounds', f'{alg_name}_Trajectory_Plot']

    # Combine all the existing arrays as necessary by separating all subgroups as unqiue groups
    if multi_group:
        num_group_types = len(agg_grouperrs)  # list of numpy arrays
        agg_grouperrs = np.column_stack(agg_grouperrs)  # vertically stck the groups errs
        if not validation:
            groupweights = np.column_stack(groupweights)  # vertically stack the weights for each groups
        stacked_group_names = []  # stack the groups errors
        for i in range(num_group_types):
            g_type = group_type[i] if num_group_types > 1 else ''
            stacked_group_names.extend([g_type + ': ' + name for name in group_names[i]])

        group_names = stacked_group_names

    # End of multi-groups adjustments

    if group_type is not None:
        # print(f'Here are the plots for groups based on: {group_type}')
        pass

    if use_input_commands and display_plots:
        input("Press `Enter` to show first plot... ")

    # Setup strings for graph titles
    dataset_string = f' on {data_name[0].upper() + data_name[1:]}'  # Set the first letter to capital if it isn't

    #plt.ion()
    # Average Pop error vs. Rounds
    figures.append(plt.figure())  # Creates figure and adds it to list of figures
    plt.plot(agg_poperrs)
    plt.title(f'Average Population Error ({pop_error_type})' + dataset_string)
    plt.xlabel('Steps')
    plt.ylabel(f'Average Population Error ({pop_error_type})')
    if display_plots:
        plt.show()

    if use_input_commands and display_plots:
        input("Next plot...")

    # Group Errors vs. Rounds
    figures.append(plt.figure())  # Create figure and append to list
    for g in range(0, len(group_names)):
        # Plots the groups with appropriate label
        plt.plot(agg_grouperrs[:, g], label=group_names[g])
    if show_legend:
        plt.legend(loc='upper right')
    plt.title(f'Group Errors ({error_type}) ' + alg_name + dataset_string)
    plt.xlabel('Steps')
    plt.ylabel(f'Group Errors ({error_type})')
    if display_plots:
        plt.show()

    if use_input_commands and display_plots:
        input("Next plot...")

    # Group Weights vs. Rounds
    if not validation and groupweights is not None:  # Groupweights aren't a part of validation
        figures.append(plt.figure())  # Create figure and append to list
        for g in range(0, len(group_names)):
            plt.plot(groupweights[:, g], label=group_names[g])
        if show_legend:
            plt.legend(loc='upper right')
        plt.title(f'Group Weights ' + alg_name + dataset_string + model_string)
        plt.xlabel('Steps')
        plt.ylabel('Group Weights')
        if display_plots:
            plt.show()

        if use_input_commands and display_plots:
            input("Next plot...")

    # Trajectory Plot with Pareto Curve
    figures.append(plt.figure())
    x = agg_poperrs
    y = np.max(agg_grouperrs, axis=1)
    points = np.zeros((len(x), 2))
    points[:, 0] = x
    points[:, 1] = y
    
    colors = np.arange(1, numsteps)
    plt.scatter(x, y, c=colors, s=2, label='Trajectory of Mixtures')
    plt.scatter(x[0], y[0], c='m', s=40, label='Starting point')  # Make the first point big and pink
    plt.title(f'Trajectory ' + alg_name + f' w/ budget {tau}' + dataset_string + model_string)
    plt.xlabel(f'Population Error ({pop_error_type})')
    plt.ylabel(f'Max Group Error ({error_type})')

    if display_plots:
        plt.show()

    for err_type, grp_errs, pop_errs, pop_err_type in bonus_plots:
        if use_input_commands and display_plots:
            input(f"Next bonus plot for error type {err_type}...")

        # Group Errors vs. Rounds
        figures.append(plt.figure())  # Create figure and append to list
        figure_names.append(f'GroupError_vs_Rounds_({err_type if err_type != "0/1 Loss" else "0-1 Loss"})')
        for g in range(0, len(group_names)):
            # Plots the groups with appropriate label
            plt.plot(grp_errs[:, g], label=group_names[g])
        if show_legend:
            plt.legend(loc='upper right')
        plt.title(f'Group Errors ({err_type})' + dataset_string + model_string)
        plt.xlabel('Steps')
        plt.ylabel(f'Group Errors ({err_type})')
        if display_plots:
            plt.show()

        if use_input_commands and display_plots:
            input("Next bonus plot (trajectory)...")

        figures.append(plt.figure())
        figure_names.append(f'Trajectory_({err_type if (err_type != "0/1 Loss") else "0-1 Loss"})')
        x = pop_errs
        y = np.max(grp_errs, axis=1)
        points = np.zeros((len(x), 2))
        points[:, 0] = x
        points[:, 1] = y

        colors = np.arange(1, numsteps)
        plt.scatter(x, y, c=colors, s=2, label='Trajectory of Mixtures')
        plt.scatter(x[0], y[0], c='m', s=40, label='Starting point')  # Make the first point big and pink
        plt.title(f'Trajectory ' + alg_name + f'with manipulation budget {tau}' + dataset_string + model_string)
        plt.xlabel(f'Population Error ({err_type})')
        plt.ylabel(f'Max Group Error ({pop_err_type})')

        if display_plots:
            plt.show()

    if use_input_commands and display_plots:
        input("Quit")

    # Update the names if doing valiadtion
    if validation:
        figure_names = [name + '_Validation' for name in figure_names]

    # Now we have a list of plots: `figures` we can save
    if save_plots:
        save_plots_to_os(figures, figure_names, dirname)
        plt.close('all')


def write_error_array(avg_error, max_error, max_error_array, avg_error_array, tau_values, dirname, val):
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(f'{dirname}', f'{val}error.txt'), "w") as file:        
        for idx, tau in enumerate(tau_values):
            file.write(f'tau = {tau} and for {val}max-error:\n') 
            file.write(f'\t Non-Strategic = {max_error[tau][0]}\n') 
            file.write(f'\t Na\u00EFve Strategic = {max_error[tau][2]}\n') 
#            file.write(f'\t NS-F = {max_error[tau][1]}\n') 
#            file.write(f'\t NS-A = {max_error[tau][3]}\n') 
            file.write(f'\t Ours = {max_error[tau][5]}\n') 

            file.write(f'tau = {tau} and for {val}avg-error:\n') 
            file.write(f'\t Non-Strategic = {avg_error[tau][0]}\n') 
            file.write(f'\t Na\u00EFve Strategic = {avg_error[tau][2]}\n') 
#            file.write(f'\t NS-F = {avg_error[tau][1]}\n') 
#            file.write(f'\t NS-A = {avg_error[tau][3]}\n') 
            file.write(f'\t Ours = {avg_error[tau][5]}\n') 

            max_error_array[idx, 0] = max_error[tau][0]
#            max_error_array[idx, 1] = max_error[tau][1]
            max_error_array[idx, 2] = max_error[tau][2]
#            max_error_array[idx, 3] = max_error[tau][3]
            max_error_array[idx, 5] = max_error[tau][5]

            avg_error_array[idx, 0] = avg_error[tau][0]
#            avg_error_array[idx, 1] = avg_error[tau][1]
            avg_error_array[idx, 2] = avg_error[tau][2]
#            avg_error_array[idx, 3] = avg_error[tau][3]
            avg_error_array[idx, 5] = avg_error[tau][5]


def pairwise_distance_distribution(X):
    # Compute the pairwise distances using pdist
    pairwise_distances = pdist(X)
    print(np.max(pairwise_distances), np.min(pairwise_distances))
    plt.hist(pairwise_distances, bins=30, edgecolor='black')
    plt.title('Histogram of Pairwise Distances')
    plt.xlabel('Pairwise Distance')
    plt.ylabel('Frequency')

    # Save the plot to a file (e.g., histogram.png)
    #plt.savefig('pairwise_distances_histogram.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()


def plot_write_overall(pop_error_type, dirname, data_name, max_error, avg_error, 
                        val_max_error, val_avg_error, tau, display_plots = False):
    dataset_string = f' on {data_name[0].upper() + data_name[1:]}'  # Set the first letter to capital if it isn't
    tau_values = list(max_error[0].keys())
    #Loop through this and then aggregate results?
    trials = len(max_error)
    max_error_array = []
    avg_error_array = []
    val_max_error_array = []
    val_avg_error_array = []
    for t in range(trials):
        max_error_array.append(np.zeros((len(tau_values),6)))
        avg_error_array.append(np.zeros((len(tau_values),6)))
        write_error_array(avg_error[t], max_error[t], max_error_array[t], avg_error_array[t], tau_values, dirname, "")    
    
        val_max_error_array.append(np.zeros((len(tau_values),6)))
        val_avg_error_array.append(np.zeros((len(tau_values),6)))
        write_error_array(val_avg_error[t], val_max_error[t], val_max_error_array[t], val_avg_error_array[t], tau_values, dirname, "val-")    
    
    #Make mean, upper, and lower arrays
    mean_max_error_array = np.mean(max_error_array, axis=0)
    max_conf = 1.96*np.std(max_error_array, axis=0)/np.sqrt(trials)

    mean_avg_error_array = np.mean(avg_error_array, axis=0)
    avg_conf = 1.96*np.std(avg_error_array, axis=0)/np.sqrt(len(avg_error_array[0]))
    mean_val_max_error_array = np.mean(val_max_error_array, axis=0)
    val_max_conf = 1.96*np.std(val_max_error_array, axis=0)/np.sqrt(len(val_max_error_array[0]))
    mean_val_avg_error_array = np.mean(val_avg_error_array, axis=0)
    val_avg_conf = 1.96*np.std(val_avg_error_array, axis=0)/np.sqrt(len(val_avg_error_array[0]))
    #May need to threshold at 0
    max_upper = mean_max_error_array + max_conf
    max_lower = mean_max_error_array - max_conf

    avg_upper = mean_avg_error_array + avg_conf
    avg_lower = mean_avg_error_array - avg_conf
    
    val_max_upper = mean_val_max_error_array + val_max_conf
    val_max_lower = mean_val_max_error_array - val_max_conf
    
    val_avg_upper = mean_val_avg_error_array + val_avg_conf
    val_avg_lower = mean_val_avg_error_array - val_avg_conf

    figures = []
    figure_names = ['MaxGroupError', 'AvgPopError', 'val_MaxGroupError', 'val_AvgPopError']
    plt.ion()
    # Average Pop error vs. Rounds
    figures.append(plt.figure())  # Creates figure and adds it to list of figures
    learner_types = ['Non-Strategic', '' , 'Na\u00EFve Strategic', '' , '' , 'Ours']
    for learner in [0, 2, 5]:
        tau_x_ticks = np.arange(0, len(mean_max_error_array[:, learner]))
        # Plots the groups with appropriate label
        plt.plot(tau_x_ticks, mean_max_error_array[:, learner], label=learner_types[learner])
        plt.fill_between(tau_x_ticks, max_upper[:, learner], max_lower[:, learner], alpha=0.1)
 
    plt.legend(loc='upper right')
    plt.title(f'Max Group (Tr) Erorr Comparison{dataset_string}')
    
    ## Changed 2 to 10
    tau_x_labels = [str(tau_values[i]) if i % 10 == 0 else '' for i in range(len(tau_values))]
    plt.xticks(tau_x_ticks, tau_x_labels)
    
    plt.xlabel('Manipulation Budget')
    plt.ylabel(f'Max Group Error ({pop_error_type})')

    if display_plots:
        plt.show()
    figures.append(plt.figure())
    for learner in [0, 2, 5]:
        # Plots the groups with appropriate label
        plt.plot(tau_x_ticks, mean_avg_error_array[:, learner], label=learner_types[learner])
        plt.fill_between(tau_x_ticks, avg_upper[:, learner], avg_lower[:, learner], alpha=0.1)
    
    plt.legend(loc='upper right')
    
   
    plt.title(f'Average Population (Tr) Error Comparison{dataset_string}')
    plt.xticks(tau_x_ticks, tau_x_labels)
    plt.xlabel('Manipulation Budget')
    plt.ylabel(f'Average Population Error ({pop_error_type})')

    if display_plots:
        plt.show()

    figures.append(plt.figure())  # Creates figure and adds it to list of figures
    for learner in [0, 2, 5]:
        # Plots the groups with appropriate label
        plt.plot(tau_x_ticks, mean_val_max_error_array[:, learner], label=learner_types[learner])
        plt.fill_between(tau_x_ticks, val_max_upper[:, learner], val_max_lower[:, learner], alpha=0.1)

 
    plt.legend(loc='upper right')
    plt.title(f'Max Group (Ts) Erorr Comparison{dataset_string}')
    
    tau_x_ticks = np.arange(0, len(mean_val_max_error_array[:, learner]))
    tau_x_labels = [str(tau_values[i]) if i % 1 == 0 else '' for i in range(len(tau_values))]
    plt.xticks(tau_x_ticks, tau_x_labels)
    
    plt.xlabel('Manipulation Budget')
    plt.ylabel(f'Max Group Error ({pop_error_type})')

    if display_plots:
        plt.show()
    
    figures.append(plt.figure())
    for learner in [0, 2, 5]:
        # Plots the groups with appropriate label
        plt.plot(tau_x_ticks, mean_val_avg_error_array[:, learner], label=learner_types[learner])
        plt.fill_between(tau_x_ticks, val_avg_upper[:, learner], val_avg_lower[:, learner], alpha=0.1)
    
    plt.legend(loc='upper right')
       
    plt.title(f'Average Population (Ts) Error Comparison{dataset_string}')
    plt.xticks(tau_x_ticks, tau_x_labels)
    plt.xlabel('Manipulation Budget')
    plt.ylabel(f'Average Population Error ({pop_error_type})')

    if display_plots:
        plt.show()

    save_plots_to_os(figures, figure_names, dirname)
    plt.close('all')
    
