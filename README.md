# Minimax Group Fairness in Strategic Classification

Code accompanying the submission “Minimax Group Fairness in Strategic Classification”.
This archive provides experiments comparing minimax fair strategic learning with
non-strategic learning and a post-processing strategic baseline. It also supports
relaxed minimax fairness experiments that trade off overall error and worst-group error.

The implementation uses a heuristic learning oracle based on shifting linear
classifiers. It is an experimental implementation, not an implementation of an
exact oracle for arbitrary hypothesis classes or manipulation costs.

## Installation

Extract the submission archive and open a terminal in its root directory.
Create a virtual environment and install the dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The dependency versions are not pinned; the archive does not include a verified
historical environment. The code uses NumPy, pandas, scikit-learn, Matplotlib,
PyTorch, and SciPy.

## Running experiments

Configure the settings near the top of `main_driver.py`, then run from the
repository root:

```sh
python3 main_driver.py
```

There is no command-line configuration interface. Important settings are:

| Setting | Purpose |
| --- | --- |
| `data_index` | Dataset: COMPAS (1), Communities (4), Credit (8), Heart (11), or synthetic data (0). |
| `model_index` | Model class; the default is the paired regression classifier (4). |
| `numsteps` | Number of learning iterations (default: 3000). |
| `num_rounds` | Number of repeated train/test splits (default: 1). |
| `random_split_seed` | Initial split seed; incremented between repetitions. |
| `tau_min`, `tau_max`, `tau_step` | Manipulation-budget sweep; inspect the construction of `tau_list` before changing the range. |
| `tau_group_values` | Group-specific budget multipliers; `None` gives equal multipliers. |
| `use_multiple_gammas`, `num_gammas` | Enable a sweep of relaxed fairness thresholds. |
| `relaxed`, `gamma` | Configure a single relaxed run when the multiple-gamma sweep is disabled. |
| `save_plots`, `save_models` | Control saved plots and models. |

The driver runs three configurations inside its experiment loop:

- `[False, False, False]`: non-strategic training without a test-time classifier shift.
- `[True, True, False]`: strategic training with strategic agents.
- `[False, False, True]`: non-strategic training followed by a test-time classifier shift.

Strategic agent responses are used for test evaluation in `src/minmaxML.py`.
Changing only the top-level `alg_spec` value does not select a single configuration;
the loop in the driver overrides it.

Results are written beneath `experiments/`, with folders identifying the dataset
and group budget multipliers. Plotting, parameter writing, and optional model
saving are implemented in `src/`. Dataset caches are written to
`vectorized_datasets/`. The default cache filename is `adults.npz`, even when a
different dataset is selected; choose a dataset-specific filename before running
if you want to preserve the bundled cache. Output settings in the driver can be
overridden by the experiment loop.

The defaults run the Heart dataset. They do not reproduce every figure in the
manuscript in one invocation. Reproducing the full experiments requires selecting
the corresponding datasets, budgets, repetitions, and fairness settings.

## Datasets

### Downloading pre-configured datasets.

To use one of the pre-configured datasets, you must first download the dataset from the corresponding online repository
, and place it in the `datasets` folder with the specified file name.

We have provided a brief description of each dataset, along with a link to a page from where it can be downloaded
. For some datasets, additional formatting or the running of provided scripts may be required.

#### Descriptions of pre-configured datasets

**Adult** - Predict whether income exceeds $50K/yr based on census data. http://archive.ics.uci.edu/ml/datasets/Adult

Downloading instructions: On the linked webpage click the "Data Folder" link and download `adult.data`. Rename the file to `adult.csv`
and place it in the `datasets` folder in this package. Then, in the `src` folder of this package, run the script
`clean_adult_data.py` which will create the the file `adult_cleaned.csv` which is formatted to be usedwith our pre-configured code.

**Seoul Bike Sharing** - The dataset contains count of public bikes rented at each hour in Seoul Bike sharing System
 with
 the
 corresponding Weather data and Holidays information. https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand
 
 Downloading instructions: On the linked webpage click the "Data Folder" link, download `SeoulBikeData.csv`,  and place
  in the `datasets` folder. If there are issues, make sure that the csv is saved in utf-8 encoding, and delete the "∞" symbol
  in the column headers "Temperature(∞C)" and "Dew point temperature(∞C)".

**Communities and Crime** - Communities within the United States. The data combines socio-economic data from the
 1990 US
 Census, law enforcement data from the 1990 US LEMAS survey, and crime data from the 1995 FBI UCR. We added a column
 for plurality race, which we used to divide communities into groups. https://github.com/algowatchpenn/GerryFair/tree/master/dataset
 
 Downloading instructions: On the provided webpage, download `communities.csv` and place in the `datasets` folder. Then, in the `src` folder of this package, run the script
   `clean_communities_data.py` which will create the the file `communities_cleaned.csv` which is formatted to be used
    with our pre-configured code.

**COMPAS** - Data from Broward County Clerk’s Office, Broward County Sherrif's Office, Florida Department of Corrections
, compiled by ProPublica. https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis

 Downloading instructions:  Go to the linked github, download the file `compas-scores-two-years.csv` and place in the `datasets`
  folder.

**Credit** - German Credit Data for predicting good and bad credit risks, compiled by Prof Dr. Hans Hoffman. https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

 Downloading instructions: Open "Data Folder" and download `german.data`. Rename to `german.csv` and place in `datasets`
  folder. Then, in the `src` folder of this package, run the script
  `setup_credit_dataset.py` which will create the the file `german_cleaned.csv` which is formatted to be used 
  with our pre-configured code.


**Default** - This dataset contains information on default payments, demographic factors, credit data, history of
 payment
, and bill statements of credit card clients in Taiwan from April 2005 to September 2005. https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

Downloading instructions: Follow the "Data Folder" link and download `default of credit card clients.xls`. Open the
 file in
 excel
 and delete the top row with X1, X2, etc. Save the file as `default.csv` and place in `datasets` folder.

**Fires** - This is a regression task, where the aim is to predict the burned area of forest fires, in the
 northeast region of Portugal, by using meteorological and other data. https://archive.ics.uci.edu/ml/datasets/forest+fires
 
Downloading instructions: In the Data Folder, download `forestfires.csv` directly into `datasets` folder.

**Heart** - This dataset contains the medical records of 299 patients who had heart failure, collected during their
 follow-up period, where each patient profile has 13 clinical features. https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records
 
 Downloading instructions: Download `heart_failure_clinical_records_dataset.csv` into `datasets` folder.

**Bank Marketing** - The data is related with direct marketing campaigns (phone calls) of a Portuguese banking
 institution. The classification goal is to predict if the client will subscribe a term deposit (variable y). https://archive.ics.uci.edu/ml/datasets/bank+marketing
 
 Downloading instructions: In the Data Folder, download `bank.zip`, extract the folder, and move the files `bank-full
 .csv` and `bank.csv` into the `datasets` folder.

**Student** - Predict student performance in secondary education (high school). https://archive.ics.uci.edu/ml/datasets/student+performance

Downloading instructions: In the Data folder, download `student.zip`, extract, and move the file `student-mat.csv` to the `datasets` folder.
If there are errors, the ";" characters may need to be replaced with ", " to meet the csv format. This can also be done by using the data-to-columns
tool in Excel and using the ";" character as a delimiter to split the first column. 

**Wine** - Using chemical analysis determine the origin of wines. https://archive.ics.uci.edu/ml/datasets/wine+quality

Downloading instructions: In the Data Folder, download `winequality-red.data`, and `winequality-white.data` and
 replace the `.data` extensions with `.csv` for both files, and place them in `datasets` folder. Then, run
  `combine_wine_data.py` which will create a new csv named `winequality-full.csv` in the `datasets` folder which is
   ready for use.

### Dataset selection and custom data

Set `use_preconfigured_dataset = True` and choose `data_index` in `main_driver.py`.
The column mappings, group definitions, and filenames are in `dataset_mapping.py`.
Some CSV datasets are bundled; download or prepare any missing files using the
instructions above. Preprocessing scripts that use `../datasets/` must be run
with `src/` as the working directory.

For custom CSV data, set `use_preconfigured_dataset = False` and configure `path`,
`label`, `groups`, `usable_features`, and `categorical_columns` in the driver.
Cached data can override these settings, so check the cache-loading section before
running a new configuration.

## Code organization

- `main_driver.py`: experiment settings, configuration loops, and aggregate plots.
- `dataset_mapping.py`: dataset filenames and feature/group mappings.
- `src/minmaxML.py`: minimax/relaxed learning and strategic evaluation.
- `src/paired_regression_classifier.py`, `src/linear_svm.py`, `src/torch_wrapper.py`: model implementations.
- Other `src/` modules: data preparation, saving, and plotting utilities.
- `datasets/`: input CSV files.
- `vectorized_datasets/`: preprocessed dataset caches.

## Third-party provenance and license notices

This code builds on the MinimaxFair implementation. The original third-party
copyright and Apache-2.0 notices are retained in source files. Some model and
dataset documentation refers to GerryFair; those references credit external
resources. Dataset use remains subject to the terms of the original providers.

## License

  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  
  Licensed under the Apache License, Version 2.0 (the "License").
  You may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  
      http://www.apache.org/licenses/LICENSE-2.0
  
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.


