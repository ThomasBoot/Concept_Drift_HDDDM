# IMPORTS

import sklearn.metrics
import pandas as pd
import numpy as np
from math import sqrt, floor
from matplotlib import pyplot as plt
from sklearn import ensemble
from Distance import Distance
from HDDDM_alternative_approach import HDDDM
from Discretize import Discretizer

# USER DEFINED SETTINGS

nr_of_batches = 500           # The amount of batches to divide the data in.
nr_bins = floor(sqrt(1000))   # The number of bins for discretization. Equals the square root of the batch cardinality.
gamma = 1.5                   # Used to define if drift is considered "large enough". See paper for more information.
warn_ratio = 0.8              # Defines at what point possible drift should be warned.
categorical_variables = []   # Defines which variables are categorical. DO NOT include target variable!
to_keep = "all"               # Variables to include in drift computation. Use statement below to specify variables.
# to_keep = categorical_variables

model = ensemble.GradientBoostingClassifier()   # Learner to be used during training
distance = Distance().hellinger_dist   # Distance metric to be used during drift detection

# LOADING THE DATASET

Dataset = 'Datasets/Hyperplane_01.csv'
Data = pd.read_csv(Dataset, delimiter=',', index_col=0)
target = 'y'    # Identify the target class

# PREPROCESSING THE DATA

Data2 = Data.copy()
discretize = Discretizer("equalquantile")  # Choose either "equalquantile" or "equalsize"
discretize.fit(Data2, None, to_ignore= categorical_variables)   # Determine which variables need discretization.
numerical_cols = discretize.numerical_cols
binned_data, bins_output = discretize.transform(Data2, nr_bins)  # Bin numerical data.

if to_keep == "all":
    detector = HDDDM(distance, categorical_variables + numerical_cols, target, gamma)
else:
    detector = HDDDM(distance, to_keep, target, gamma)

Batch = np.array_split(Data, nr_of_batches)  # Batching the dataset.

# INITIAL MODEL TRAINING

X_train = Batch[0].iloc[:, 0:-1]
y_train = Batch[0].iloc[:, -1]

model.fit(X_train, y_train)

# ITERATING THROUGH THE BATCHES + DRIFT DETECTION

drift = []      # Stores detected drifts
warning = []    # Stores detected warnings prior to drift.
accuracy = []   # Stores model accuracy per batch.
magnitude = []  # Stores the magnitude of change between batches.

Drift = np.array_split(binned_data, nr_of_batches)  # Always use the discretized data for drift detection!
Drift_ref = Drift[0]

for i in range(1, nr_of_batches):
    X_batch = Batch[i].iloc[:, 0:-1]
    y_batch = Batch[i].iloc[:, -1]
    y_pred = model.predict(X_batch)
    Drift_batch = Drift[i]

    acc = sklearn.metrics.accuracy_score(y_batch, y_pred)
    accuracy.append(acc)

    detector.update(Drift_ref, Drift_batch, warn_ratio)
    drift_magnitude = detector.windows_distance(Drift_ref, Drift_batch)
    magnitude.append(drift_magnitude)

    if detector.detected_warning_zone():
        warning.append(i)

    elif detector.detected_change():
        drift.append(i)
        Drift_ref = Drift_batch  # Reset the batch to be used as reference.
        # print(f'Drift detected in batch {i} with drift magnitude {drift_magnitude}')
        detector.reset()

        model = model.fit(X_batch, y_batch)  # Retrain the model

    else:
        Drift_ref = pd.concat([Drift_ref, Drift_batch])  # Extend the reference batch.

# print(f'\nOverview of Detected warnings: {warning}')
print(f'\nOverview of Detected drifts in batches: {drift}')
# print(f'\nOverview of Distance magnitudes: {magnitude}')


# VISUALIZING OUTPUT


def visualize_drift(acc, drift, warn):
    fig, ax = plt.subplots()
    x = np.linspace(nr_of_batches, len(X_train)*nr_of_batches, num = nr_of_batches - 1)
    ax.set_xlabel('index')
    ax.set_ylabel('accuracy')

    plt.vlines(x=[value*len(X_train) for value in warn], ymin=0, ymax=1, colors='g', linestyles=':', label='warnings')
    plt.vlines(x=[value*len(X_train) for value in drift], ymin=0, ymax=1, colors='r', linestyles='-', label='drifts')
    ax.plot(x, acc, lw=2, label='accuracy')

    ax.legend()
    plt.title("Hyperplane - High (10%) Gradual Drift")
    plt.show()
    fig.savefig('Figures/Hyperplane_01_2.png')


def visualize_magnitude(magn):
    fig, ax = plt.subplots()
    ax.plot(magn, label = "drift magnitude")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Magnitude")

    ax.legend()
    plt.title("Hyperplane - High (10%) Gradual Drift")
    plt.show()
    fig.savefig('Figures/Hyperplane_01_magn_2.png')


visualize_drift(accuracy, drift, warning)
visualize_magnitude(magnitude)
