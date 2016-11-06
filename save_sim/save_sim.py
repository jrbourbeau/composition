#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v1/icetray-start
#METAPROJECT /data/user/jbourbeau/metaprojects/icerec/V05-00-00/build

import numpy as np
import time
import glob
import argparse
import os

from icecube import dataio, toprec, dataclasses, icetray, phys_services
from icecube.frame_object_diff.segments import uncompress
from I3Tray import *
from icecube.tableio import I3TableWriter
from icecube.hdfwriter import I3HDFTableService
from icecube.icetop_Level3_scripts.functions import count_stations

import composition.support_functions.simfunctions as simfunctions
import composition.support_functions.i3modules as i3modules


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description='Runs extra modules over a given fileList')
    p.add_argument('-f', '--files', dest='files', nargs='*',
                   help='Files to run over')
    p.add_argument('-s', '--sim', dest='sim',
                   help='Simulation dataset')
    p.add_argument('-o', '--outfile', dest='outfile',
                   help='Output file')
    args = p.parse_args()

    # Starting parameters
    IT_pulses, inice_pulses = simfunctions.reco_pulses()

    # Keys to write to frame
    keys = []
    keys += ['I3EventHeader']
    keys += ['ShowerPlane']
    # keys += ['ShowerPlane', 'ShowerPlaneParams']
    keys += ['ShowerCOG']
    keys += ['MCPrimary']
    keys += ['IceTopMaxSignal', 'IceTopMaxSignalString',
             'IceTopMaxSignalInEdge', 'IceTopNeighbourMaxSignal',
             'StationDensity', 'NStations']
    keys += ['NChannels', 'InIce_charge', 'max_charge_frac']
    # keys += ['NChannels_CoincPulses', 'InIce_charge_CoincPulses',
    #          'NChannels_SRTCoincPulses', 'InIce_charge_SRTCoincPulses']
    keys += ['InIce_FractionContainment', 'IceTop_FractionContainment',
             'LineFit_InIce_FractionContainment']
    keys += ['LaputopParams']

    t0 = time.time()

    # Construct list of non-truncated files to process
    good_file_list = []
    for test_file in args.files:
        try:
            test_tray = I3Tray()
            test_tray.context['I3FileStager'] = dataio.get_stagers(
                staging_directory=os.environ['_CONDOR_SCRATCH_DIR'])
            test_tray.Add('I3Reader', FileName=test_file)
            test_tray.Add(uncompress, 'uncompress')
            test_tray.Execute()
            test_tray.Finish()
            good_file_list.append(test_file)
        except:
            print('file {} is truncated'.format(test_file))
            pass
    del test_tray

    tray = I3Tray()
    tray.context['I3FileStager'] = dataio.get_stagers(
        staging_directory=os.environ['_CONDOR_SCRATCH_DIR'])
    tray.Add('I3Reader', FileNameList=good_file_list)
    # Uncompress Level3 diff files
    tray.Add(uncompress, 'uncompress')
    hdf = I3HDFTableService(args.outfile)

    # Filter out non-coincident frames
    tray.Add(lambda frame: frame['IceTopInIce_StandardFilter'].value)

    def get_nstations(frame):
        nstation = 0
        if IT_pulses in frame:
            nstation = count_stations(
                dataclasses.I3RecoPulseSeriesMap.from_frame(frame, IT_pulses))
        frame.Put('NStations', icetray.I3Int(nstation))

    tray.Add(get_nstations)

    # def get_inice_charge(frame):
    #     q_tot = 0.0
    #     n_channels = 0
    #     if inice_pulses in frame:
    #         VEMpulses = frame[inice_pulses]
    #         if VEMpulses.__class__ == dataclasses.I3RecoPulseSeriesMapMask:
    #             VEMpulses = VEMpulses.apply(frame)
    #
    #             for om, pulses in VEMpulses:
    #                 n_channels += 1
    #                 for pulse in pulses:
    #                     q_tot += pulse.charge
    #
    #     frame.Put('InIce_charge', dataclasses.I3Double(q_tot))
    #     frame.Put('NChannels', icetray.I3Int(n_channels))
    #     return

    # tray.Add(get_inice_charge)

    # Add total inice charge to frame
    tray.Add(i3modules.AddInIceCharge, inice_pulses='SRTCoincPulses')

    # Add containment to frame
    tray.Add(i3modules.AddMCContainment)
    tray.Add(i3modules.AddInIceRecoContainment)

    #====================================================================
    # Finish

    tray.Add(I3TableWriter, tableservice=hdf, keys=keys,
             SubEventStreams=['ice_top'])

    tray.Execute()
    tray.Finish()

    print('Time taken: {}'.format(time.time() - t0))
