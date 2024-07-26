# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import numpy as np
import os
from src.save_plots import save_plots_to_os


def do_plotting(display_plots, save_plots, use_input_commands, numsteps, group_names, group_type,
                show_legend, error_type, data_name, model_string,
                agg_poperrs, agg_grouperrs, groupweights, pop_error_type, bonus_plots,
                dirname, tau, strategic_learner, multi_group=False,
                validation=False, equal_error=False):
    """
    Helper function for minimaxML that creates the relevant plots for a single run of the simulation.
    """

    # Create a list of all figures we want to save for later which will be passed into a function

    str_learner ={True: 'S', False: 'NS'}
    
    strategic_string = f'of {str_learner[strategic_learner[0]]}-{str_learner[strategic_learner[1]]} learner'

    figures = []
    learner_name = f'{str_learner[strategic_learner[0]]}-{str_learner[strategic_learner[1]]} learner'
    figure_names = [f'{learner_name}_PopError_vs_Rounds', f'{learner_name}_GroupError_vs_Rounds', f'{learner_name}_GroupWeights_vs_Rounds', f'{learner_name}_Trajectory_Plot']

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
    plt.title(f'Group Errors ({error_type}) ' + strategic_string + dataset_string)
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
        plt.title(f'Group Weights ' + strategic_string + dataset_string + model_string)
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
    plt.title(f'Trajectory ' + strategic_string + f' w/ budget {tau}' + dataset_string + model_string)
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
        plt.title(f'Trajectory ' + strategic_string + f'with manipulation budget {tau}' + dataset_string + model_string)
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


def plot_write_overall(pop_error_type, dirname, data_name, max_error, avg_error, val_max_error, val_avg_error, tau, display_plots = False):
    dataset_string = f' on {data_name[0].upper() + data_name[1:]}'  # Set the first letter to capital if it isn't
    tau_values = list(max_error.keys())
    max_error_array = np.zeros((len(tau_values),4))
    avg_error_array = np.zeros((len(tau_values),4))
    val_max_error_array = np.zeros((len(tau_values),4))
    val_avg_error_array = np.zeros((len(tau_values),4))
     
    
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(f'{dirname}', "error_stat.txt"), "w") as file:        
        for idx, tau in enumerate(tau_values):
            file.write(f'tau = {tau} and for max-error:\n') 
            file.write(f'\t with non-strategic learner (training) and non-strategic (test) = {max_error[tau][0]}\n') 
            file.write(f'\t with non-strategic learner (training) and strategic (test) = {max_error[tau][1]}\n') 
            file.write(f'\t with strategic learner (training) and strategic (test) = {max_error[tau][3]}\n') 

            
            file.write(f'tau = {tau} and for avg-error:\n') 
            file.write(f'\t with non-strategic learner (training) and non-strategic (test) = {avg_error[tau][0]}\n') 
            file.write(f'\t with non-strategic learner (training) and strategic (test) = {avg_error[tau][1]}\n') 
            file.write(f'\t with strategic learner (training) and strategic (test) = {avg_error[tau][3]}\n') 

            max_error_array[idx, 0] = max_error[tau][0]
            max_error_array[idx, 1] = max_error[tau][1]
            max_error_array[idx, 3] = max_error[tau][3]

            avg_error_array[idx, 0] = avg_error[tau][0]
            avg_error_array[idx, 1] = avg_error[tau][1]
            avg_error_array[idx, 3] = avg_error[tau][3]

    with open(os.path.join(f'{dirname}', "val_error_stat.txt"), "w") as file:        
        for idx, tau in enumerate(tau_values):
            file.write(f'tau = {tau} and for val_max-error:\n') 
            file.write(f'\t with non-strategic learner (training) and non-strategic (test) = {val_max_error[tau][0]}\n') 
            file.write(f'\t with non-strategic learner (training) and strategic (test) = {val_max_error[tau][1]}\n') 
            file.write(f'\t with strategic learner (training) and strategic (test) = {val_max_error[tau][3]}\n') 

            
            file.write(f'tau = {tau} and for val_avg-error:\n') 
            file.write(f'\t with non-strategic learner (training) and non-strategic (test) = {val_avg_error[tau][0]}\n') 
            file.write(f'\t with non-strategic learner (training) and strategic (test) = {val_avg_error[tau][1]}\n') 
            file.write(f'\t with strategic learner (training) and strategic (test) = {val_avg_error[tau][3]}\n') 

            val_max_error_array[idx, 0] = val_max_error[tau][0]
            val_max_error_array[idx, 1] = val_max_error[tau][1]
            val_max_error_array[idx, 3] = val_max_error[tau][3]

            val_avg_error_array[idx, 0] = val_avg_error[tau][0]
            val_avg_error_array[idx, 1] = val_avg_error[tau][1]
            val_avg_error_array[idx, 3] = val_avg_error[tau][3]
    
     
    figures = []
    figure_names = ['MaxGroupError', 'AvgPopError', 'val_MaxGroupError', 'val_AvgPopError']
    plt.ion()
    # Average Pop error vs. Rounds
    figures.append(plt.figure())  # Creates figure and adds it to list of figures
    learner_types = ['NS-NS', 'NS-S', '' , 'S-S']
    for learner in [0, 1, 3]:
        # Plots the groups with appropriate label
        plt.plot(max_error_array[:, learner], label=learner_types[learner])
 
    plt.legend(loc='upper right')
    plt.title(f'NS-NS vs NS-S vs S-S Learner' + dataset_string)
    
    tau_x_ticks = np.arange(0, len(max_error_array[:, learner]))
    tau_x_labels = [str(tau_values[i]) if i % 2 == 0 else '' for i in range(len(tau_values))]
    plt.xticks(tau_x_ticks, tau_x_labels)
    
    plt.xlabel('Manipulation Budget')
    plt.ylabel(f'Max Group Error ({pop_error_type})')

    if display_plots:
        plt.show()
    
    figures.append(plt.figure())
    for learner in [0, 1, 3]:
        # Plots the groups with appropriate label
        plt.plot(avg_error_array[:, learner], label=learner_types[learner])
    
    plt.legend(loc='upper right')
    
   
    plt.title(f'NS-NS vs NS-S vs S-S Learner' + dataset_string)
    plt.xticks(tau_x_ticks, tau_x_labels)
    plt.xlabel('Manipulation Budget')
    plt.ylabel(f'Average Population Error ({pop_error_type})')

    if display_plots:
        plt.show()

    figures.append(plt.figure())  # Creates figure and adds it to list of figures
    for learner in [0, 1, 3]:
        # Plots the groups with appropriate label
        plt.plot(val_max_error_array[:, learner], label=learner_types[learner])
 
    plt.legend(loc='upper right')
    plt.title(f'NS-NS vs NS-S vs S-S Learner' + dataset_string)
    
    tau_x_ticks = np.arange(0, len(val_max_error_array[:, learner]))
    tau_x_labels = [str(tau_values[i]) if i % 2 == 0 else '' for i in range(len(tau_values))]
    plt.xticks(tau_x_ticks, tau_x_labels)
    
    plt.xlabel('Manipulation Budget')
    plt.ylabel(f'Max Group Error ({pop_error_type})')

    if display_plots:
        plt.show()
    
    figures.append(plt.figure())
    for learner in [0, 1, 3]:
        # Plots the groups with appropriate label
        plt.plot(val_avg_error_array[:, learner], label=learner_types[learner])
    
    plt.legend(loc='upper right')
    
   
    plt.title(f'NS-NS vs NS-S vs S-S Learner' + dataset_string)
    plt.xticks(tau_x_ticks, tau_x_labels)
    plt.xlabel('Manipulation Budget')
    plt.ylabel(f'Average Population Error ({pop_error_type})')

    if display_plots:
        plt.show()


    save_plots_to_os(figures, figure_names, dirname)
    plt.close('all')
    