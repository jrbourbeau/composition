#!/usr/bin/env python

from __future__ import division
import numpy as np
import pandas as pd
import time
import glob
import re
from collections import OrderedDict

import composition.support_functions.paths as paths


def load_sim(config='IT73', bintype='logdist', return_cut_dict=False):

    # Load simulation dataframe
    mypaths = paths.Paths()
    infile = '{}/{}_sim/sim_dataframe.hdf5'.format(
        mypaths.comp_data_dir, config, bintype)

    df = pd.read_hdf(infile)

    # Quality Cuts #
    # Adapted from PHYSICAL REVIEW D 88, 042004 (2013)
    cut_dict = OrderedDict()
    # IT specific cuts
    cut_dict['reco_exists'] = df['reco_exists']
    # cut_dict['reco_passed'] = (df['reco_energy'] > 0)
    # cut_dict['reco_zenith'] = (np.cos(df['ShowerPlane_zenith']) >= 0.8)
    cut_dict['reco_zenith'] = (np.cos(np.pi - df['reco_zenith']) >= 0.8)
    cut_dict['reco_IT_containment'] = (df['reco_IT_containment'] <= 1.0)
    cut_dict['IceTopMaxSignalInEdge'] = np.logical_not(
        df['IceTopMaxSignalInEdge'].astype(bool))
    cut_dict['IceTopMaxSignal'] = (df['IceTopMaxSignal'] >= 6)
    # InIce specific cuts
    cut_dict['NChannels'] = (df['NChannels'] > 0)
    cut_dict['InIce_containment'] = (df['InIce_FractionContainment'] <= 1.0)
    cut_dict['reco_InIce_containment'] = (df['reco_IT_containment'] <= 1.0)

    selection_mask = np.array([True] * len(df))
    standard_cut_keys = ['reco_exists', 'reco_zenith', 'reco_IT_containment',
                         'IceTopMaxSignalInEdge', 'IceTopMaxSignal', 'NChannels', 'reco_InIce_containment']
    for key in standard_cut_keys:
        selection_mask *= cut_dict[key]

    # Add log-energy and log-charge columns to df
    df['MC_log_energy'] = np.nan_to_num(np.log10(df['MC_energy']))
    df['reco_log_energy'] = np.nan_to_num(np.log10(df['reco_energy']))
    df['InIce_log_charge'] = np.nan_to_num(np.log10(df['InIce_charge']))
    df['reco_cos_zenith'] = np.cos(np.pi - df['reco_zenith'])
    df['ShowerPlane_cos_zenith'] = np.cos(df['ShowerPlane_zenith'])
    df['log_s125'] = np.log10(df['s125'])

    # # Adapted versions of Bakhtiyar's cuts
    # c['llh1'] = np.logical_not(np.isnan(s['ML_energy']))
    # c['llh2'] = (np.cos(np.pi - s['zenith']) >= 0.8)
    # # Get rid of ML_containment nan's
    # s['ML_containment'][np.isnan(s['ML_containment'])] = 1000.
    # c['llh3'] = (s['ML_containment'] <= 1.0)
    # c['llh4'] = np.logical_not(s['IceTopMaxSignalInEdge'])
    # c['llh5'] = (np.nan_to_num(s['IceTopMaxSignal']) >= 6)
    # # Final versions of cuts for testing
    # c['llh'] = c['llh1'] * c['llh2'] * c['llh3'] * c['llh4'] * c['llh5']
    # s['cuts'] = c

    # Print cut event flow
    n_total = len(df)
    cut_eff = {}
    cumulative_cut_mask = np.array([True] * n_total)
    # cut_labels = ['reco_passed', 'reco_zenith', 'reco_IceTop_containment',
    #               'IceTopMaxSignalInEdge', 'IceTopMaxSignal', 'reco_InIce_containment']
    print('Cut event flow:')
    for cut_label, mask in cut_dict.iteritems():
        cumulative_cut_mask *= mask
        print('{:>30}:  {:>5.3}  {:>5.3}'.format(cut_label, np.sum(
            mask) / n_total, np.sum(cumulative_cut_mask) / n_total))
    print('\n')

    if return_cut_dict:
        return df, cut_dict
    else:
        return df[selection_mask]
