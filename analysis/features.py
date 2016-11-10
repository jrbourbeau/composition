#!/usr/bin/env python

import numpy as np

def get_training_features():
    # feature_list = np.array(['reco_log_energy', 'InIce_log_charge'])
    feature_list = np.array(['reco_log_energy', 'InIce_log_charge', 'reco_cos_zenith'])

    return feature_list
