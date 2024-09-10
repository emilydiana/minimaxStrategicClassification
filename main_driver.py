# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from src.generate_matrices import generate_synthetic_data
from src.setup_matrices import setup_matrices
from src.minmaxML import do_learning
from src.read_file import read_dataset_from_file
from dataset_mapping import get_dataset_features
from src.plot_relaxed_pareto import do_pareto_plot
from src.write_params_to_file import write_params_to_os
from collections import defaultdict
from src.plotting import plot_write_overall
import numpy as np
import random
import os
import warnings

strategic_learner = [False, False] #strategic_learner[0]: is learner strategic in training, #strategic_learner[1]: is strategic in test
strategic_agent = [False, True] 
alg_spec = [False, False, False]    # the first two coordinates are respectively showing type of the learner and agents in the Tr phase, 
                                    # and the last coordinate indicates learner behavior in the Ts phase.
                                    # In other words, alg_spec assign values to strategic_learner and strategic_agent as follows:
                                    # strategic_learner[0] = alg_spec[0]
                                    # strategic_learner[1] = alg_spec[2]    
                                    # strategic_agent = alg_spec[1]       
tau_min = 0
tau_max = 1
tau_step = 0.5
num_rounds = 2 
decimal_size = 1

#scale: COMAPS =1, Credit = 1, Student = 1e9


#Good Parameters for COMAPS: tau = 0.5 & tau = 1 (all most all points), tau= 0.2 & tau = 0.1 ( tn is close to fp), tau=0.25 & tau= 0.3 are good
#Good Parameters for Communities: tau = 0.1 (contains all points), tau = 0.01 is good



# MODEL/SIMULATION Settings
models = {1: 'LinearRegression', 2: 'LogisticRegression', 3: 'Perceptron', 4: 'PairedRegressionClassifier',
          5: 'MLPClassifier', 6: 'LinearSVM'}  # WARNING: MLPClassifier is not GPU optimized and may run slowly
model_index = 4  # Set this to select a model type according to the mapping above

numsteps = 3000  # number of steps for learning/game
# NOTE: eta = a * t^(-b) on the t-th round of the game
a = 1  # Multiplicative coefficient on parametrized learning rate
b = 1 / 2  # Negative exponent on parameterized learning rate
scale_eta_by_label_range = True  # Multiplies `a` by square of max abs. label value, to 'normalize' regression labels
equal_error = False  # Defaults to False for minimax. Set to True to find equal error solution
error_type = '0/1 Loss'  # 'MSE', '0/1 Loss', 'FP', 'FN', 'Log-Loss', 'FP-Log-Loss', 'FN-Log-Loss'
extra_error_types = {}  # Set of additional error types to plot from (only relevant for classification)
pop_error_type = ''  # Error type for the population on the trajectory (set automatically in general)
test_size = 0.3  # The proportion of the training data to be withheld as validation data (set to 0.0 for no validation)
random_split_seed = 4235255  # If test_string1 size > 0.0, the seed to be passed to numpy for train/test split

fit_intercept = True  # If the linear model should fit an intercept (applies only to LinReg and Logreg)
convergence_threshold = 1e-12  # Converge early if max change in sampleweights between rounds is less than threshold

# Relaxed Model Settings
use_multiple_gammas = True  # Set to True to run relaxed algo over many values of gamma
num_gammas = 2  # If use_multiple_games, number of intermediate gammas to use between min and max feasible gamma
# Use these arguments to run a single relaxed simulation with on gamma settting
relaxed = False  # Determines if single run
gamma = 0.0  # Max groups error if using relaxed variant


# Solver Specific Settings

# Settings for Logistic Regression (if used)
logistic_solver = 'liblinear'  # Which logistic regression solver algorithm to use from sklearn (liblinear recommended)
tol = 1e-15  # The tolerance on the gradient for logistic regression convergence, sklearn default is 1-e4
max_logi_iters = 100000  # Maximum iterations of logistic regression algorithm
penalty = 'l2'  # Regularization penalty for log-loss: 'none', 'l1', 'l2'
C = 1e15  # Inverse of regularization strength, ignored when penalty = 'none'. Set to 1e15 to simulate no regularization

# Settings for Multi-Layer Perceptron (if used)
# NOTE: Current implementation uses ReLU for all hidden layers and sigmoid for output layer
n_epochs = 1000
lr = 0.1
momentum = 0.9
weight_decay = 0
# Hidden sizes is a list denoting the size of each hidden layer in the MLP. Fractional values in the list represent
# proportions of the input layer, and whole numbers represent absolute layer sizes.
hidden_sizes = [0.5]


# Dataset Settings

# Settings for reading and formatting a dataset from a csv
path = ''  # Path to csv file within 'datasets' folder (which should be in root directory along with this python file)
label = ''  # String denoting the column name of the label for prediction
groups = []  # List of one or more strings denoting the column names on which to form groups (e.g. race, gender)
usable_features = []  # List of column names of the features we want to use for prediction
categorical_columns = []
groups_to_drop = []
is_categorical = True  # Denotes whether labels are categeorical (classification) or numeric (regression)

# Pre-structured datasets -- IGNORED if `use_preconfigured_dataset` is False
use_preconfigured_dataset = True  # Set to True and select a data_index to use an existing dataset (or synthetic data)
datasets = {1: 'COMPAS', 2: 'COMPAS_full', 3: 'Default', 4: 'Communities', 5: 'Adult', 6: 'Student',
            7: 'Bike', 8: 'Credit', 9: 'Fires', 10: 'Wine', 11: 'Heart', 12: 'Marketing(Small)', 13: 'Marketing(Full)',
            14: 'COMPAS_race_and_gender', 15: 'Student_binary',
            0: 'Synthetic'}
#Good Datasets: COMPAS, 
#Adult does not converge. But still good for showing the difference

# 1, 4, , 8, 11
 
data_index = 1  # Set this to select a dataset by index according to the mapping above (0 for synthetic)
drop_group_as_feature = True  # Set to False (default) if groups should also be a one hot encoded categorical feature

# Data read/write settings
read_from_file = False  # If we should read pre-computed numpy matrices from a file - OVERRIDES the above if set to True
save_data = True  # Whether or not data from setting up matrices should be is saved to the specified directory
file_dir = 'vectorized_datasets'  # Directory for files containing vectorized datasets to read from/write to
file_name = 'adults.npz'  # File name within file_dir from which to read or write data, should be .npz file
file_path = os.path.join(file_dir, file_name)

# Synthetic Data Settings  (NOTE: only used if data_index = 0 and use_preconfigured_dataset = True)
numsamples = 200  # number of instances/rows of X
numdims = 10  # dimensionality of synthetic data
noise = 1  # gaussian noise in y
# Group features
num_group_types = 1  # num groups 'types' (e.g. race, sex) such that each instance belongs to some subgroup of each type
min_subgroups = 2  # min number of subgroups for each groups type
max_subgroups = 4  # max number of subgroups for each groups type
min_subgroup_size = max(numdims, 30)  # NOTE: this needs to be <= 50 for most reasonable sized nsamples <10k with
use_new_seed_each_run = True  # Enable this to randomly generate a random seed, rather than using the fixed one
random_data_seed = 8890956  # If use_new_seed_each_run is False, random seed to be used in for synthetic data generation
# Settings for the numerical values of each feature for each groups
mean_range = 0  # Mean for each feature is selected from [-mean_range, mean_range], be careful making this too large
variability = 1  # Variability which is std. dev for normal features and is distance from mean to endpoints for uniform
num_uniform_features = 0  # How many of the "numdims" features should be uniform (the rest are normally distributed
intercept_scale = 2  # Coefficient on randomly generated` intercept for each groups  (0.0 means no intercept)

# Plot/output settings
verbose = False  # enables verbose output for doLearning
display_plots = False 
display_intermediate_plots = False # Whether or not to display intermediate plots in during relaxation
use_basic_plots = False  # Whether or not we want to save/display the simple gamma vs error plots
show_legend = True  # Denotes if the plots show the legend 
use_input_commands = False  # Enables 'input(next)' to delay plots until entry into terminal
dirname = 'auto-Results'  # Specifies which directory to save to, recommend to prefix with 'auto-'
# NOTE: Use dirname == '' or 'auto-<OUTER DIRECTORY>' to use automatically generated inner folder name

# Data saving settings
save_plots = True  # If True, saves plots as PNGs to `dirname` directory (recommended since plots dissapear otherwise)
save_intermediate_plots = True  # Relevant for plot_multiple_gammas, saves intermediate plots to file
save_models = False  # (MEMORY INTENSIVE: not recommended) saves models to `dirname` directory and returns them as list


# ----------------------------------------------CODE PROCESSING ---------------------------------------
if __name__ == '__main__':

    # Input arguments
    #arg1 = sys.argv[1]
    #arg2 = sys.argv[2]

    # Define this list for later
    classification_models = ['LogisticRegression', 'Perceptron', 'PairedRegressionClassifier',
                             'MLPClassifier', 'LinearSVM']

    new_synthetic = False  # Needed for automatic data naming later

    # Select the random seed for synthetic data generation as a random value or the specified value
    random_data_seed = random_data_seed if not use_new_seed_each_run else random.randint(0, 10000000)

    # If we are reading from a file, let the dataset be named by the file it's saved to, minus the .npz extension
    data_name = file_name[:-4] if read_from_file else datasets[data_index]
    if data_name == 'Synthetic':
        data_name += f'_{random_data_seed}'

    # If we have already cached the dataset as a .npz, don't need to read from csv and can instead read directly
    if os.path.isfile(os.path.join('vectorized_datasets', data_name + '.npz')):
        print(f'We found a cached version of this dataset ({data_name}). Using cached version...')
        read_from_file = True
        file_path = os.path.join('vectorized_datasets', data_name + '.npz')

    # Get the information about the dataset based on the index selected above, unless it's saved already
    if not read_from_file:
        if use_preconfigured_dataset:
            path, label, groups, usable_features, categorical_columns, groups_to_drop, is_categorical \
                = get_dataset_features(datasets[data_index])
    else:
        warnings.warn('WARNING: read_from_file is True, so other dataset settings will be ignored and matrices will '
                      'be read directly from specified file. If this was not intended, please set read_from_file to '
                      'False.')
        path, label, groups, usable_features, categorical_columns, groups_to_drop, is_categorical \
            = None, None, None, None, None, None, None

    if path is not '' and path is not None and not path.startswith('datasets/'):
        path = 'datasets/' + path

    new_synthetic = path is '' and not read_from_file
    model_type = models[model_index]

    binary = model_type in classification_models  # If synthetic data, create binary data if using a classifier


    # Setup matrices from data from file
    if read_from_file:
        X, y, grouplabels, group_names, group_types, is_categorical = read_dataset_from_file(file_path)
    else:  # Create matrices corresponding to given input path and features
        group_types = []
        if new_synthetic:
            X, y, grouplabels, group_names, group_types, is_categorical = \
                generate_synthetic_data(numdims, noise, numsamples, num_group_types,
                                        min_subgroups=min_subgroups, max_subgroups=max_subgroups,
                                        min_subgroup_size=min_subgroup_size,
                                        mean_range=mean_range, variability=variability, intercept_scale=intercept_scale,
                                        num_uniform_features=num_uniform_features,
                                        binary=binary,
                                        save_data=save_data, file_dir=file_dir, file_name=file_name,
                                        random_seed=random_data_seed, drop_group_as_feature=drop_group_as_feature)
        else:
            X, y, grouplabels, group_names, group_types, is_categorical = \
                setup_matrices(path, label, groups, usable_features=usable_features,
                               drop_group_as_feature=drop_group_as_feature,
                               categorical_columns=categorical_columns, groups_to_drop=groups_to_drop,
                               verbose=verbose,
                               save_data=save_data, file_dir=file_dir, file_name=file_name)
    if model_type in classification_models:
        if not is_categorical:
            raise Exception('You selected a classifier with a non-categorical dataset.')
    else:
        if is_categorical and data_index != 0:
            warnings.warn('WARNING: You selected a regression model with categorically labeled data. '
                          'Consider using a different model type')

    # Allows us to give shorter names to our folders
    model_name_shortener = {'PairedRegressionClassifier': 'PRC', 'LinearRegression': 'LinReg',
                            'LogisticRegression': 'LogReg', 'LinearSVM': 'SVM'}
    tau_list = [round(i * tau_step, 1) for i in range(tau_min, tau_max + 1)]
    # tau_group_values indicate the fraction of tau values each group is using. 
    # group i budget is tau_group_values[i] * tau
    tau_group_values = np.ones(np.array(group_names).flatten().shape, dtype=float)
    
    avg_error = []
    max_error = []
    val_avg_error = []
    val_max_error = []
    #Emily hacky correction for upping seed each round
    random_split_seed -=1
    for t in range(num_rounds):
        random_split_seed +=1
        avg_error.append(defaultdict(lambda: [0] * 6))
        max_error.append(defaultdict(lambda: [0] * 6))
        val_avg_error.append(defaultdict(lambda: [0] * 6))
        val_max_error.append(defaultdict(lambda: [0] * 6))
        for tau in tau_list:                            
            curr_avg_error = [0] * 6 
            curr_max_error = [0] * 6
            curr_val_avg_error = [0] * 6
            curr_val_max_error = [0] * 6
        
            # Initialize tau vector. Different groups with different tau values        
            flattened_grouplabels = grouplabels.flatten()
            tau_vector = (tau_group_values[flattened_grouplabels]) * tau
            # Initially, we set the learner_tau equal to the mean value. However, we want to learn this value too. 
            learner_tau_mean = np.mean(tau_vector)
            learner_tau_min_frac = np.min(tau_group_values)
            learner_tau_max_frac = np.max(tau_group_values)    

            dataname_extension = data_name if not new_synthetic else f'seed={random_data_seed}'
            outer_directory = f'experiments/{dataname_extension}'
            error_tag = '_' + (error_type if error_type != '0/1 Loss' else '0-1 Loss')
            equal_error_tag = '_equal-error' if equal_error else ''
            solver_tag = f'_{logistic_solver}' if model_type == 'LogisticRegression' else ''
            model_tag = model_name_shortener.get(model_type, model_type)
            dirname = f'{outer_directory}/{model_tag}_val={test_size}_tau={round(tau,decimal_size)}{error_tag}{equal_error_tag}'
        
            for alg_spec in [[False, False, False], [True, True, False], [False, False, True]]:
                # Assign values from alg_spec to strategic_learner and strategic_agent
                strategic_learner[0] = alg_spec[0]  # Is learner strategic in training?
                strategic_learner[1] = alg_spec[2]  # Does learner shift its classifier in test time?   
                strategic_agent = alg_spec[1]       # Are agents strategic in training time?
                curr_index = strategic_learner[0] * 4 + strategic_learner[1] * 2 + strategic_agent

                if not use_multiple_gammas:
                    print(f'Executing main with the following parameters: \n \n\
                    model: {model_type} \n \
                    dataset: {data_name}\n \
                    numrounds: {numsteps}\n \
                    a: {a}\n \
                    b: {b}\n \
                    test_size = {test_size}\n \
                    error_type: {error_type}')
                    print('relaxed:', relaxed)

                    if model_type == 'LogisticRegression':
                        print('fit_intercept:', fit_intercept)
                        print('solver:', logistic_solver)
                        print('max_iterations:', max_logi_iters)
                        print('tol:', tol)
                    if relaxed:
                        print('gamma:', gamma)
                    if test_size > 0.0:
                        print('random_split_seed:', random_split_seed)
                    if new_synthetic:
                        print('numdims:', numdims)
                        print('gaussian noise in y:', noise)
                        print('numsamples:', numsamples)
                        print('random_data_seed:', random_data_seed)

                    do_learning(X, y, numsteps, grouplabels, a, b, equal_error=equal_error,
                                scale_eta_by_label_range=scale_eta_by_label_range, model_type=model_type,
                                group_names=group_names, group_types=group_types, data_name=data_name,
                                gamma=gamma, relaxed=relaxed, random_split_seed=random_split_seed,
                                verbose=verbose, use_input_commands=use_input_commands,
                                error_type=error_type, extra_error_types=extra_error_types, pop_error_type=pop_error_type,
                                convergence_threshold=convergence_threshold,
                                show_legend=show_legend, save_models=save_models,
                                display_plots=display_plots,
                                test_size=test_size,
                                fit_intercept=fit_intercept, logistic_solver=logistic_solver,
                                max_logi_iters=max_logi_iters, tol=tol, penalty=penalty, C=C,
                                n_epochs=n_epochs, lr=lr, momentum=momentum, weight_decay=weight_decay, hidden_sizes=hidden_sizes,
                                save_plots=save_plots, dirname=dirname, 
                                strategic_learner=strategic_learner, strategic_agent=strategic_agent, 
                                tau=tau, tau_vector=tau_vector,
                                learner_tau_min_frac = learner_tau_min_frac, learner_tau_max_frac = learner_tau_max_frac, learner_tau_mean = learner_tau_mean,
                                learner_tau_step = 0.1, curr_idx = curr_index,
                                max_error=curr_max_error, avg_error=curr_avg_error, 
                                val_max_error=curr_val_max_error, val_avg_error=curr_val_avg_error)
                    max_error[t][tau][curr_index] = curr_max_error[curr_index]
                    avg_error[t][tau][curr_index] = curr_avg_error[curr_index]
                    val_max_error[t][tau][curr_index] = curr_val_max_error[curr_index]
                    val_avg_error[t][tau][curr_index] = curr_val_avg_error[curr_index]

                # If we do the relaxed version of the code, use an unrelaxed simulation to find the bounds on gamma
                else:
                    print('Starting a multi-round relaxed simulation over many values of gamma.')
                    print('To run a single simulation, set `plot_multiple_gammas` to False')
                    print(f'Here are the baseline parameters: \n \n \
                    model: {model_type} \n \
                    dataset: {data_name}\n \
                    num_rounds: {numsteps}\n \
                    num_gammas: {num_gammas}\n \
                    a: {a}\n \
                    b: {b}\n \
                    test_size = {test_size}\n \
                    error_type: {error_type}')

                    if model_type == 'LogisticRegression':
                        print('fit_intercept:', fit_intercept)
                        print('solver:', logistic_solver)
                        print('max_iterations:', max_logi_iters)
                        print('tol:', tol)
                        print()

                    if test_size > 0.0:
                        print('random_split_seed:', random_split_seed)
                    if new_synthetic:
                        # print('numgroups:', numgroups)
                        print('numdims:', numdims)
                        print('gaussian noise in y:', noise)
                        print('numsamples:', numsamples)
                        print('random_data_seed:', random_data_seed)
                        print()

                    if error_type in ['MSE', '0/1 Loss', 'Log-Loss']:
                        minimax_err, max_err, initial_pop_err, agg_grouperrs, agg_poperrs, _, pop_err_type, total_steps, _, _, _, \
                        _, _, _ = \
                            do_learning(X, y, numsteps, grouplabels, a, b, equal_error=False,
                                        scale_eta_by_label_range=scale_eta_by_label_range, model_type=model_type,
                                        gamma=0.0, relaxed=False, random_split_seed=random_split_seed,
                                        group_names=group_names, group_types=group_types, data_name=data_name,
                                        verbose=verbose, use_input_commands=use_input_commands,
                                        error_type=error_type, extra_error_types=extra_error_types, pop_error_type=pop_error_type,
                                        convergence_threshold=convergence_threshold,
                                        show_legend=show_legend, save_models=False,
                                        display_plots=display_intermediate_plots,
                                        test_size=test_size, fit_intercept=fit_intercept, logistic_solver=logistic_solver,
                                        max_logi_iters=max_logi_iters, tol=tol, penalty=penalty, C=C,
                                        n_epochs=n_epochs, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                        hidden_sizes=hidden_sizes,
                                        save_plots=save_intermediate_plots, dirname=dirname,
                                        strategic_learner=strategic_learner, strategic_agent=strategic_agent, 
                                        tau=tau, tau_vector=tau_vector,
                                        learner_tau_min_frac = learner_tau_min_frac, learner_tau_max_frac = learner_tau_max_frac, learner_tau_mean = learner_tau_mean,
                                        learner_tau_step = 0.1, curr_idx = curr_index,
                                        max_error=curr_max_error, avg_error=curr_avg_error, 
                                        val_max_error=curr_val_max_error, val_avg_error=curr_val_avg_error)
                        print(f'With our non-relaxed simulation, we found the range of feasible gammas to be ' +
                              f'[{minimax_err}, {max_err}]')
                    # In the FP, FN case, simply try all values between 0 and the groups error achieved with min pop error
                    # Instead of accepting a max error rate of 1, we only need to accept the max error rate when pop error
                    # is minimized (this may not be exactly true due to heuristic nature of classification)
                    elif error_type in ['FP', 'FN', 'FP-Log-Loss', 'FN-Log-Loss']:
                        numrounds = 1 if not equal_error else numsteps
                        disp_plots = equal_error  # If we are in equal error case, then we display plots
                        verb = verbose and equal_error

                        # Run a single run to find the max groups error when pop error is minimized
                        minimax_err, max_err, initial_pop_err, _, _, _, pop_err_type, total_steps, _, _, _, _, _, _ = \
                            do_learning(X, y, numrounds, grouplabels, a, b, equal_error=equal_error,
                                        scale_eta_by_label_range=scale_eta_by_label_range, model_type=model_type,
                                        fit_intercept=fit_intercept, logistic_solver=logistic_solver,
                                        convergence_threshold=convergence_threshold,
                                        max_logi_iters=max_logi_iters, tol=tol, penalty=penalty, C=C,
                                        n_epochs=n_epochs, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                        hidden_sizes=hidden_sizes,
                                        gamma=0.0, relaxed=False, random_split_seed=random_split_seed,
                                        group_names=group_names, group_types=group_types, data_name=data_name,
                                        verbose=verb, use_input_commands=False,
                                        error_type=error_type, display_plots=disp_plots, test_size=test_size)
                        if not equal_error:
                            # We can always drive FP/FN rates to 0 by always predicting negative/positive
                            minimax_err = 0
                            print(f'We found the range of feasible gammas to be [{minimax_err}, {max_err}]')
                        else:
                            print(f'We will try the single value for gamma = {minimax_err}')

                    else:
                        raise Exception(f'Invalid error type: {error_type}')

                    # Write parameters to file
                    params_list = [f'model_index = {model_index}', f'error_type = {error_type}', f'numsteps = {numsteps}', f'a = {a}',
                                   f'b = {b}',
                                   f'scale_eta_by_label_range = {scale_eta_by_label_range}', f'test_size = {test_size}',
                                   f'fit_intercept={fit_intercept}', f'tol={tol}', f'logistic_solver={logistic_solver}',
                                   f'max_logi_iters = {max_logi_iters}',
                                   f'random_split_seed = {random_split_seed}',
                                   f'use_multiple_gammas = {use_multiple_gammas}', f'num_gammas = {num_gammas}', f'relaxed = {relaxed}',
                                   f'gamma = {gamma if relaxed else 0.0}',
                                   f'data_index = {data_index}', f'drop_group_as_feature = {drop_group_as_feature}',
                                   f'tau_min = {tau_min}', f'tau_max = {tau_max}', f'tau_step = {tau_step}',
                                   f'tau_group_values = {tau_group_values}']
                    gammas = []
                    #total_steps_per_gamma = []  # need to track the length until convergence for each run individually
                    #max_grp_errs = []
                    #pop_errs = []
                    #trajectories = []
                    #bonus_plot_list = []

                    #val_max_grp_errs = []
                    #val_pop_errs = []
                    #val_trajectories = []
                    #val_bonus_plot_list = []
                        
                    max_error[t][tau][curr_index]=[]
                    avg_error[t][tau][curr_index]=[]
                    val_max_error[t][tau][curr_index]=[]
                    val_avg_error[t][tau][curr_index]=[]

                    increment = (max_err - minimax_err) / num_gammas  # NOTE: `max_err` is defined over all subgroups
                    if increment == 0:
                        assert max_err == minimax_err  # this should be the only way increment is 0
                        warnings.warn(f'WARNING: Range of feasible gammas consists of only 1 value: {minimax_err}.'
                                      ' Running a single simulation with this value...')
                        gamma_list = [minimax_err]

                    else:
                        gamma_list = np.arange(minimax_err, max_err + increment, increment)

                    # If we are using equal error, then we only care about minimax s.t. all errors below that
                    if equal_error:
                        gamma_list = [minimax_err]

                    # Perform one iteration for each value of gamma
                    for gamma in gamma_list:
                        # Skip gammas that are unnecessarily loose as a result of rounding while including endpoint
                        if gamma > max_err and len(gamma_list) > 1:
                            print(f'Skipping overly loose gamma value: {gamma}')
                            continue
                        print(f'Starting relaxed learning with gamma = {gamma}...')
                        (max_grp_err, _, _, agg_grouperrs, agg_poperrs, bonus_plots, pop_err_type, total_steps, _,
                         val_grp_err, val_pop_err, val_agg_grouperrs, val_agg_poperrs, val_bonus_plots) = \
                            do_learning(X, y, numsteps, grouplabels, a, b, equal_error=False,
                                        scale_eta_by_label_range=scale_eta_by_label_range, model_type=model_type,
                                        gamma=gamma, relaxed=True,
                                        random_split_seed=random_split_seed,
                                        group_names=group_names, group_types=group_types, data_name=data_name,
                                        verbose=verbose, use_input_commands=use_input_commands,
                                        error_type=error_type, extra_error_types=extra_error_types, pop_error_type=pop_error_type,
                                        convergence_threshold=convergence_threshold,
                                        show_legend=show_legend, save_models=save_models,
                                        display_plots=display_intermediate_plots,
                                        test_size=test_size, fit_intercept=fit_intercept, logistic_solver=logistic_solver,
                                        max_logi_iters=max_logi_iters, tol=tol, penalty=penalty, C=C,
                                        n_epochs=n_epochs, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                        hidden_sizes=hidden_sizes,
                                        save_plots=save_intermediate_plots, dirname=dirname + f'/Gamma={gamma}/',
                                        strategic_learner=strategic_learner, strategic_agent=strategic_agent, 
                                        tau=tau, tau_vector=tau_vector,
                                        learner_tau_min_frac = learner_tau_min_frac, learner_tau_max_frac = learner_tau_max_frac, learner_tau_mean = learner_tau_mean,
                                        learner_tau_step = 0.1, curr_idx = curr_index,
                                        max_error=curr_max_error, avg_error=curr_avg_error, 
                                        val_max_error=curr_val_max_error, val_avg_error=curr_val_avg_error)
                        # Max groups errors and pop errors of the final mixture for a pareto curve
                        #Leave this alone for strat class
                        gammas.append(gamma)
                        #Don't plot this for strat class
                        #total_steps_per_gamma.append(total_steps)
                        #Change form of these next two arrays
                        max_error[t][tau][curr_index].append(curr_max_error[curr_index])
                        avg_error[t][tau][curr_index].append(curr_avg_error[curr_index])
                        val_max_error[t][tau][curr_index].append(curr_val_max_error[curr_index])
                        val_avg_error[t][tau][curr_index].append(curr_val_avg_error[curr_index])

                        #max_grp_errs.append(max_grp_err)
                        #pop_errs.append(agg_poperrs[-1])

                        # Stack the grouperrs across all groups types and then make trajectories
                        #agg_grouperrs_stacked = np.column_stack(agg_grouperrs)
                        #xs = agg_poperrs
                        #ys = np.max(agg_grouperrs_stacked, axis=1)
                        #trajectories.append((xs, ys))

                        # NOTE: bonus plots are "stacked" bonus plots (i.e. we stack all subgroups across grouptypes)
                        #bonus_plot_list.append(bonus_plots)

                        #Update this for strat class
                        #if test_size > 0.0:
                        #    val_max_grp_errs.append(val_grp_err)
                        #    val_pop_errs.append(val_agg_poperrs[-1])
                        #    val_agg_grouperrs_stacked = np.column_stack(val_agg_grouperrs)
                        #    val_x = val_agg_poperrs
                        #    val_y = np.max(val_agg_grouperrs_stacked, axis=1)
                        #    val_trajectories.append((val_x, val_y))
                        #    val_bonus_plot_list.append(val_bonus_plots)

                    # End of relaxed simulations over all gammas

                # Write parameters to file
                params_list = [f'model_index = {model_index}', f'error_type = {error_type}', f'numsteps = {numsteps}', f'a = {a}',
                               f'b = {b}',
                               f'scale_eta_by_label_range = {scale_eta_by_label_range}', f'test_size = {test_size}',
                               f'fit_intercept={fit_intercept}', f'tol={tol}', f'logistic_solver={logistic_solver}',
                               f'max_logi_iters = {max_logi_iters}',
                               f'random_split_seed = {random_split_seed}',
                               f'use_multiple_gammas = {use_multiple_gammas}', f'num_gammas = {num_gammas}', f'relaxed = {relaxed}',
                               f'gamma = {gamma if relaxed else 0.0}',
                               f'data_index = {data_index}', f'drop_group_as_feature = {drop_group_as_feature}',
                               f'tau_min = {tau_min}', f'tau_max = {tau_max}', f'tau_step = {tau_step}']

                synethetic_list = []
                if use_preconfigured_dataset and data_index == 0 and not read_from_file:
                    params_list.extend([f'numsamples = {numsamples}', f'num_group_types = {num_group_types}',
                                        f'numdims = {numdims}', f'noise = {noise}',
                                        f'random_data_seed = {random_data_seed}',
                                        f'mean_range = {mean_range}',
                                        f'variability = {variability}',
                                        f'intercept_scale = {intercept_scale}',
                                        f'num_uniform_features = {num_uniform_features}'])

                write_params_to_os(outer_directory, params_list)
    if use_multiple_gammas:
        do_pareto_plot(gammas, max_error, avg_error,
                   error_type, pop_err_type,
                   save_plots, dirname,
                   model_type,
                   use_input_commands,
                   data_name=data_name, show_basic_plots=use_basic_plots,
                   val_max_grp_errs=val_max_error, val_pop_errs=val_avg_error,
                   test_size=test_size)
    plot_write_overall(error_type, outer_directory, data_name, max_error, avg_error, 
                       val_max_error, val_avg_error, tau, display_plots=False)

