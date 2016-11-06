#!/usr/bin/env python

import numpu as np

def get_features()
    # feature_list = np.array(['reco_log_energy', 'reco_cos_zenith', 'InIce_log_charge',
    #                          'NChannels', 'NStations', 'reco_radius', 'reco_InIce_containment', 'log_s125'])
    feature_list = np.array(['InIce_log_charge', 'reco_InIce_containment', 'NChannels'])

    return feature_list
