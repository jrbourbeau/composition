#!/usr/bin/env python

import glob
import re
import os
import argparse
import time
import getpass

import composition.support_functions.paths as paths
import composition.support_functions.simfunctions as simfunctions
from composition.support_functions.dag_submitter import py_submit
from composition.support_functions.checkdir import checkdir


def get_argdict(comp_data_dir, **args):

    argdict = dict.fromkeys(args['sim'])
    for sim in args['sim']:

        arglist = []

        # Get config and simulation files
        config = simfunctions.sim2cfg(sim)
        gcd, files = simfunctions.get_level3_files(sim)

        # Default parameters
        outdir = '{}/{}_sim/files'.format(comp_data_dir, config)
        checkdir(outdir + '/')
        if args['test']:
            args['n'] = 2

        # List of existing files to possibly check against
        existing_files = glob.glob('{}/sim_{}_*.hdf5'.format(outdir, sim))
        existing_files.sort()

        # Split into batches
        batches = [files[i:i + args['n']]
                   for i in range(0, len(files), args['n'])]
        if args['test']:
            batches = batches[:2]

        for bi, batch in enumerate(batches):

            # Name output hdf5 file
            start_index = batch[0].find('Run') + 3
            end_index = batch[0].find('.i3.gz')
            start = batch[0][start_index:end_index]
            end = batch[-1][start_index:end_index]
            out = '{}/sim_{}_part{}-{}.hdf5'.format(outdir, sim, start, end)

            # Don't forget to insert GCD file at beginning of FileNameList
            batch.insert(0, gcd)
            batch = ' '.join(batch)

            arg = '--files {} -s {} -o {}'.format(batch, sim, out)

            arglist.append(arg)

        argdict[sim] = arglist

    return argdict


def get_merge_argdict(**args):

    # Build arglist for condor submission
    merge_argdict = dict.fromkeys(args['sim'])
    for sim in args['sim']:
        merge_args = '-s {}'.format(sim)
        if args['overwrite']:
            merge_args += ' --overwrite'
        if args['remove']:
            merge_args += ' --remove'
        merge_argdict[sim] = merge_args

    return merge_argdict


def make_submit_script(executable, jobID, script_path, condordir):

    checkdir(script_path)

    lines = ["universe = vanilla\n",
             "getenv = true\n",
             "executable = {}\n".format(executable),
             "arguments = $(ARGS)\n",
             "log = /scratch/{}/logs/{}.log\n".format(
                 getpass.getuser(), jobID),
             "output = {}/outs/{}.out\n".format(condordir, jobID),
             "error = {}/errors/{}.error\n".format(condordir, jobID),
             "notification = Never\n",
             # "+AccountingGroup=\"long.$ENV(USER)\"\n",
             "queue \n"]

    condor_script = script_path
    with open(condor_script, 'w') as f:
        f.writelines(lines)

    return


def getjobID(jobID, comp_data_dir):
    jobID += time.strftime('_%Y%m%d')
    othersubmits = glob.glob(
        '{}/condor/submit_scripts/{}_??.submit'.format(comp_data_dir, jobID))
    jobID += '_{:02d}'.format(len(othersubmits) + 1)
    return jobID

if __name__ == "__main__":

    # Setup global path names
    mypaths = paths.Paths()
    checkdir(mypaths.comp_data_dir)
    simoutput = simfunctions.getSimOutput()
    default_sim_list = ['7006', '7579', '7241', '7263', '7791',
                        '7242', '7262', '7851', '7007', '7784']

    p = argparse.ArgumentParser(
        description='Runs save_sim.py on cluster en masse',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=simoutput)
    p.add_argument('-s', '--sim', dest='sim', nargs='*',
                   choices=default_sim_list,
                   default=default_sim_list,
                   help='Simulation to run over')
    p.add_argument('-n', '--n', dest='n', type=int,
                   default=500,
                   help='Number of files to run per batch')
    p.add_argument('--test', dest='test', action='store_true',
                   default=False,
                   help='Option for running test off cluster')
    p.add_argument('--maxjobs', dest='maxjobs', type=int,
                   default=3000,
                   help='Maximum number of jobs to run at a given time.')
    p.add_argument('--overwrite', dest='overwrite',
                   default=False, action='store_true',
                   help='Overwrite existing merged files')
    p.add_argument('--remove', dest='remove',
                   default=False, action='store_true',
                   help='Remove unmerged hdf5 files')
    args = p.parse_args()

    cwd = os.getcwd()
    jobID = 'save_sim'
    jobID = getjobID(jobID, mypaths.comp_data_dir)
    cmd = '{}/save_sim.py'.format(cwd)
    argdict = get_argdict(mypaths.comp_data_dir, **vars(args))
    condor_script = '{}/condor/submit_scripts/{}.submit'.format(
        mypaths.comp_data_dir, jobID)
    make_submit_script(cmd, jobID, condor_script,
        mypaths.comp_data_dir + '/condor')

    merge_jobID = 'merge_sim'
    merge_jobID = getjobID(merge_jobID, mypaths.comp_data_dir)
    merge_cmd = '{}/merge.py'.format(cwd)
    merge_argdict = get_merge_argdict(**vars(args))
    merge_condor_script = '{}/condor/submit_scripts/{}.submit'.format(
        mypaths.comp_data_dir, merge_jobID)
    make_submit_script(merge_cmd, merge_jobID, merge_condor_script,
                       mypaths.comp_data_dir + '/condor')

    # Set up dag file
    jobID = 'save_sim_merge'
    jobID = getjobID(jobID, mypaths.comp_data_dir)
    dag_file = '{}/condor/submit_scripts/{}.submit'.format(
        mypaths.comp_data_dir, jobID)
    checkdir(dag_file)
    with open(dag_file, 'w') as dag:
        for sim in argdict.keys():
            parent_string = 'Parent '
            if len(argdict[sim]) < 1:
                continue
            for i, arg in enumerate(argdict[sim]):
                dag.write('JOB sim_{}_p{} '.format(sim, i) +
                          condor_script + '\n')
                dag.write('VARS sim_{}_p{} '.format(sim, i) +
                          'ARGS="' + arg + '"\n')
                parent_string += 'sim_{}_p{} '.format(sim, i)
            dag.write('JOB merge_{} '.format(
                sim) + merge_condor_script + '\n')
            dag.write('VARS merge_{} '.format(sim) +
                      'ARGS="' + merge_argdict[sim] + '"\n')
            child_string = 'Child merge_{}'.format(sim)
            dag.write(parent_string + child_string + '\n')

    # Submit jobs
    os.system('condor_submit_dag -maxjobs {} {}'.format(args.maxjobs, dag_file))
