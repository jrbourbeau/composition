#!/usr/bin/env python

##############################################################################
# Load essential information for simulation analysis
##############################################################################

import numpy as np
import time
import glob
import re

import composition.support_functions.paths as paths
# import LLH_tools


def load_sim(config='IT73', bintype='logdist'):

    # Load simulation information
    mypaths = paths.Paths()
    infile = '{}/{}_sim/sim_dataframe.hdf5'.format(
        mypaths.comp_data_dir, config, bintype)

    t0 = time.time()
    print('Working on {}...'.format(infile))

    df = pd.read_hdf(infile)

    ## Quality Cuts ##
    c = {}

    c['IceTopMaxSignal'] = (np.nan_to_num(s['IceTopMaxSignal']) >= 6)
    c['MC_zenith'] = (np.cos(np.pi - s['MC_zenith']) >= 0.8)
    c['InIce_FractionContainment'] = (s['InIce_FractionContainment']) <= 1.0)
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

    # # Print
    # n_total = float(len(s['eventID']))
    # cut_eff = {}
    # cumulative_cut_mask = np.array([True] * len(s['eventID']))
    # cut_labels = ['reconstructed', 'zenith', 'containment',
    #               'loudest edge', 'max charge']
    # print('Cut event flow:')
    # for i, label in zip(range(1, 6), cut_labels):
    #     cut_mask = np.array([True] * len(s['eventID']))
    #     cut_mask *= s['cuts']['llh{}'.format(i)]
    #     cumulative_cut_mask *= s['cuts']['llh{}'.format(i)]
    #     print('{:>15}:  {:>5.3}  {:>5.3}'.format(label, np.sum(
    #         cut_mask) / n_total, np.sum(cumulative_cut_mask) / n_total))
    #
    # return s

if __name__ == "__main__":

    s = load_sim()
