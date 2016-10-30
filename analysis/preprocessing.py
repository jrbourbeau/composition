#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, train_test_split
# Import classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def get_train_test_sets(df):
    # Load and preprocess training data
    feature_list = ['ML_energy', 'MC_zenith', 'InIce_charge']
    X, y = df[feature_list].values , df.comp.values
    # Convert comp string labels to numerical labels
    y = LabelEncoder().fit_transform(y)

    # Split data into training and test samples
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=2)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # Scale features and labels
    # NOTE: the scaler is fit only to the training features
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    return X_train_std, X_test_std, y_train, y_test
