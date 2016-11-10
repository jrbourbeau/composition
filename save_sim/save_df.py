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

    p = argparse.ArgumentParser(
        description='Runs extra modules over a given fileList')
    p.add_argument('-o', '--outfile', dest='outfile',
                   help='Output file')
    args = p.parse_args()

    dataframe_dict = defaultdict(list)

    # Get simulation information
    t_sim = time.time()
    print('Loading simulation information...')
    file_list = sorted(glob.glob(mypaths.comp_data_dir +
                                 '/IT73_sim/files/sim_????.hdf5'))
    value_keys = ['IceTopMaxSignal',
                  'IceTopMaxSignalInEdge',
                  'IceTopMaxSignalString',
                  'IceTopNeighbourMaxSignal',
                  'InIce_charge',
                  'NChannels',
                  'max_charge_frac',
                  'NStations',
                  'StationDensity',
                  'IceTop_FractionContainment',
                  'InIce_FractionContainment']
    for f in file_list:
        print('\tWorking on {}'.format(f))
        sim_dict = {}
        store = pd.HDFStore(f)
        for key in value_keys:
            sim_dict[key] = store.select(key).value
        # Get MCPrimary information
        for key in ['x', 'y', 'energy', 'zenith', 'azimuth', 'type']:
            sim_dict['MC_{}'.format(key)] = store.select('MCPrimary')[key]
        # Get s125
        sim_dict['s125'] = store.select('LaputopParams')['s125']
        # Get ShowerPlane zenith reconstruction
        sim_dict['ShowerPlane_zenith'] = store.select('ShowerPlane').zenith
        # Add simulation set number and corresponding composition
        sim_num = os.path.splitext(f)[0].split('_')[-1]
        sim_dict['sim'] = np.array([sim_num] * len(store.select('MCPrimary')))
        sim_dict['MC_comp'] = np.array(
            [simfunctions.sim2comp(sim_num)] * len(store.select('MCPrimary')))
        store.close()
        for key in sim_dict.keys():
            dataframe_dict[key] += sim_dict[key].tolist()
    print('Time taken: {}'.format(time.time() - t_sim))
    print('Time per file: {}\n'.format((time.time() - t_sim) / 4))

    # Get ShowerLLH reconstruction information
    t_LLH = time.time()
    print('Loading ShowerLLH reconstructions...')
    file_list = sorted(glob.glob(mypaths.llh_dir +
                                 '/IT73_sim/files/SimLLH_????_logdist.hdf5'))
    for f in file_list:
        print('\tWorking on {}'.format(f))
        LLH_dict = {}
        store = pd.HDFStore(f)
        # Get most-likely composition
        LLH_particle = store.select('ShowerLLH')
        LLH_dict['reco_exists'] = LLH_particle.exists.astype(bool)
        # Get ML energy
        LLH_dict['reco_energy'] = LLH_particle.energy
        # Get ML core position
        LLH_dict['reco_x'] = LLH_particle.x
        LLH_dict['reco_y'] = LLH_particle.y
        # Get ML core radius
        LLH_dict['reco_radius'] = np.sqrt(
            LLH_dict['reco_x']**2 + LLH_dict['reco_y']**2)
        # Get ML zenith
        LLH_dict['reco_zenith'] = LLH_particle.zenith
        # Get ShowerLLH containment information
        LLH_dict['reco_IT_containment'] = store.select(
            'ShowerLLH_IceTop_containment').value
        LLH_dict['reco_InIce_containment'] = store.select(
            'ShowerLLH_InIce_containment').value
        # Get ShowerLLH+lap hybrid containment information
        LLH_dict[
            'LLH-lap_IT_containment'] = store.select('LLH-lap_IceTop_containment').value
        LLH_dict[
            'LLH-lap_InIce_containment'] = store.select('LLH-lap_InIce_containment').value
        LLH_dict['combined_reco_exists'] = store.select(
            'LLH-lap_InIce_containment').exists.astype(bool)

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
