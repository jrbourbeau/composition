#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v1/icetray-start
#METAPROJECT /data/user/jbourbeau/metaprojects/icerec/V05-00-00/build

from __future__ import division
import numpy as np
import pandas as pd
import time
import glob
import argparse
import os
from collections import defaultdict

import composition.support_functions.simfunctions as simfunctions
import composition.support_functions.paths as paths
from composition.support_functions.checkdir import checkdir

if __name__ == "__main__":
    # Setup global path names
    mypaths = paths.Paths()
    checkdir(mypaths.comp_data_dir)
    simoutput = simfunctions.getSimOutput()
    default_sim_list = ['7006', '7579', '7007', '7784']

    p = argparse.ArgumentParser(
        description='Runs extra modules over a given fileList')
    p.add_argument('-s', '--sim', dest='sim', nargs='*',
                   choices=default_sim_list,
                   default=default_sim_list,
                   help='Simulation to run over')
    p.add_argument('-o', '--outfile', dest='outfile',
                   help='Output file')
    args = p.parse_args()

    dataframe_dict = defaultdict(list)
    # Get simulation information
    t_sim = time.time()
    print('Loading simulation information')
    file_list = glob.glob(mypaths.comp_data_dir +
                          '/IT73_sim/files/sim_????.hdf5')
    value_keys = ['IceTopMaxSignal',
                  'IceTopMaxSignalInEdge',
                  'IceTopMaxSignalString',
                  'InIce_FractionContainment',
                  'InIce_charge',
                  'NChannels',
                  'StationDensity']
    for f in file_list:
        print('\tWorking on {}...'.format(f))
        sim_dict = {}
        store = pd.HDFStore(f)
        for key in value_keys:
            sim_dict[key] = store.select(key).value
        # Get MCPrimary information
        for key in ['x', 'y', 'energy', 'zenith', 'azimuth', 'type']:
            sim_dict['MC_{}'.format(key)] = store.select('MCPrimary')[key]
        # Add simulation set number
        sim_num = os.path.splitext(f)[0].split('_')[-1]
        sim_dict['sim'] = np.array([sim_num] * len(store.select('MCPrimary')))
        store.close()
        for key in sim_dict.keys():
            dataframe_dict[key] += sim_dict[key].tolist()
    print('Time taken: {}'.format(time.time() - t_sim))
    print('Time per file: {}'.format((time.time() - t_sim) / 4))

    # Get ShowerLLH reconstruction information
    t_LLH = time.time()
    print('Loading ShowerLLH reconstructions')
    file_list = glob.glob(mypaths.comp_data_dir +
                          '/IT73_sim/files/showerLLH_????_*.hdf5')
    value_keys = ['maxLLH']
    for f in file_list:
        print('Working on {}...'.format(f))
        LLH_dict = {}
        store = pd.HDFStore(f)
        proton_maxLLH = store.select('ShowerLLHParams_proton').maxLLH
        iron_maxLLH = store.select('ShowerLLHParams_iron').maxLLH
        LLH_array = np.array([proton_maxLLH, iron_maxLLH]).T
        maxLLH_index = np.argmax(LLH_array, axis=1)
        showerLLH_proton = store.select('ShowerLLH_proton')
        showerLLH_iron = store.select('ShowerLLH_iron')
        # Get ML energy
        energy_choices = np.array(
            [showerLLH_proton.energy, showerLLH_iron.energy]).T
        ML_energy = np.choose(maxLLH_index, energy_choices)
        LLH_dict['ML_energy'] = ML_energy
        # Get ML core position
        x_choices = np.array(
            [showerLLH_proton.x, showerLLH_iron.x]).T
        ML_x = np.choose(maxLLH_index, x_choices)
        LLH_dict['ML_x'] = ML_x
        y_choices = np.array(
            [showerLLH_proton.y, showerLLH_iron.y]).T
        ML_y = np.choose(maxLLH_index, y_choices)
        LLH_dict['ML_y'] = ML_y
        store.close()

        for key in LLH_dict.keys():
            dataframe_dict[key] += LLH_dict[key].tolist()

    print('Time taken: {}'.format(time.time() - t_LLH))
    print('Time per file: {}'.format((time.time() - t_LLH) / 4))
    
    # Convert value lists to arrays (faster than using np.append?)
    for key in dataframe_dict.keys():
        dataframe_dict[key] = np.asarray(dataframe_dict[key])

    df = pd.DataFrame.from_dict(dataframe_dict)
    df.to_hdf('{}/IT73_sim/sim_dataframe.hdf5'.format(mypaths.comp_data_dir),
              'dataframe', mode='w')
