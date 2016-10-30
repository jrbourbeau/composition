#!/usr/bin/env python

from __future__ import division
from collections import Counter
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.apionly as sns
from mlxtend.evaluate import plot_decision_regions

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier

from icecube import ShowerLLH

from composition.analysis.load_sim import load_sim


if __name__ == '__main__':

    sns.set_palette('muted')
    sns.set_color_codes()

    p = argparse.ArgumentParser(
        description='Runs extra modules over a given fileList')
    p.add_argument('-e', dest='energy',
                   choices=['MC', 'reco'],
                   default='MC',
                   help='Output directory')
    p.add_argument('--outdir', dest='outdir',
                   default='/home/jbourbeau/public_html/figures/composition',
                   help='Output directory')
    args = p.parse_args()

    '''Throughout this code, X will represent features,
       while y will represent class labels'''

    df = load_sim()
    # Preprocess training data
    # feature_list = np.array(['reco_log_energy', 'reco_cos_zenith',
    # 'InIce_log_charge',
    feature_list = np.array(['reco_log_energy', 'reco_cos_zenith', 'InIce_log_charge',
                             'NChannels', 'NStations', 'reco_radius', 'reco_InIce_containment', 'log_s125'])
    for feature in feature_list:
        print('var {} = {}'.format(feature, df[feature].var()))
    X, y = df[np.append(feature_list, 'MC_log_energy')
              ].values, df.MC_comp.values
    # Convert comp string labels to numerical labels
    le = LabelEncoder().fit(y)
    y = le.transform(y)

    # Split data into training and test samples
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=2)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # Scale features and labels
    # NOTE: the scaler is fit only to the training features
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    # X_train_std = X_train_std[:, :-1]
    X_test_std = stdsc.transform(X_test)
    # X_test_std = X_test_std[:, :-1]

    print('events = ' + str(y_train.shape[0]))

    forest = RandomForestClassifier(
        n_estimators=500, max_depth=6, criterion='gini', max_features=None, n_jobs=-1, verbose=1)
    # Train forest on training data
    forest.fit(X_train_std[:, :-1], y_train)
    name = forest.__class__.__name__
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    print('=' * 30)
    print(name)
    test_predictions = forest.predict(X_test_std[:, :-1])
    test_acc = accuracy_score(y_test, test_predictions)
    print('Test accuracy: {:.4%}'.format(test_acc))
    train_predictions = forest.predict(X_train_std[:, :-1])
    train_acc = accuracy_score(y_train, train_predictions)
    print('Train accuracy: {:.4%}'.format(train_acc))
    print('=' * 30)

    reco_comp = le.inverse_transform(train_predictions)
    correctly_identified_mask = (train_predictions == y_train)

    LLH_bins = ShowerLLH.LLHBins(bintype='logdist')
    energy_bins = LLH_bins.bins['E']
    energy_bins = energy_bins[energy_bins >= 6.2]
    energy_midpoints = (energy_bins[1:] + energy_bins[:-1]) / 2
    if args.energy == 'MC':
        log_energy = stdsc.inverse_transform(X_train_std)[:, -1]
    if args.energy == 'reco':
        log_energy = stdsc.inverse_transform(X_train_std)[:, 0]
    MC_proton_mask = (le.inverse_transform(y_train) == 'P')
    MC_iron_mask = (le.inverse_transform(y_train) == 'Fe')
    # Get number of MC proton and iron as a function of MC energy
    num_MC_protons_energy = np.histogram(log_energy[MC_proton_mask],
                                         bins=energy_bins)[0]
    num_MC_protons_energy_err = np.sqrt(num_MC_protons_energy)
    num_MC_irons_energy = np.histogram(log_energy[MC_iron_mask],
                                       bins=energy_bins)[0]
    num_MC_irons_energy_err = np.sqrt(num_MC_irons_energy)
    num_MC_total_energy = np.histogram(log_energy, bins=energy_bins)[0]
    num_MC_total_energy_err = np.sqrt(num_MC_total_energy)

    # Get number of reco proton and iron as a function of MC energy
    num_reco_proton_energy = np.histogram(
        log_energy[MC_proton_mask & correctly_identified_mask],
        bins=energy_bins)[0]
    num_reco_proton_energy_err = np.sqrt(num_reco_proton_energy)
    num_reco_iron_energy = np.histogram(
        log_energy[MC_iron_mask & correctly_identified_mask],
        bins=energy_bins)[0]
    num_reco_iron_energy_err = np.sqrt(num_reco_iron_energy)
    num_reco_total_energy = np.histogram(
        log_energy[correctly_identified_mask],
        bins=energy_bins)[0]
    num_reco_total_energy_err = np.sqrt(num_reco_total_energy)

    # Calculate reco proton and iron fractions as a function of MC energy
    reco_proton_frac = num_reco_proton_energy / num_MC_protons_energy
    reco_proton_frac_err = reco_proton_frac * np.sqrt(
        ((num_reco_proton_energy_err) / (num_reco_proton_energy))**2 +
        ((num_MC_protons_energy_err) / (num_MC_protons_energy))**2)

    reco_iron_frac = num_reco_iron_energy / num_MC_irons_energy
    print(num_MC_irons_energy)
    print(num_reco_iron_energy)
    print(reco_proton_frac)
    reco_iron_frac_err = reco_iron_frac * np.sqrt(
        ((num_reco_iron_energy_err) / (num_reco_iron_energy))**2 +
        ((num_MC_irons_energy_err / num_MC_irons_energy)**2))

    reco_total_frac = num_reco_total_energy / num_MC_total_energy
    reco_total_frac_err = reco_total_frac * np.sqrt(
        ((num_reco_total_energy_err) / (num_reco_total_energy))**2 +
        ((num_MC_total_energy_err) / (num_MC_total_energy))**2)

    # Plot fraction of events vs energy
    fig, ax = plt.subplots()
    ax.errorbar(energy_midpoints, reco_proton_frac,
                yerr=reco_proton_frac_err,
                marker='.', markersize=10,
                label='Proton')
    ax.errorbar(energy_midpoints, reco_iron_frac,
                yerr=reco_iron_frac_err,
                marker='.', markersize=10,
                label='Iron')
    ax.errorbar(energy_midpoints, reco_total_frac,
                yerr=reco_total_frac_err,
                marker='.', markersize=10,
                label='Total')
    ax.axhline(0.5, linestyle='-.', marker='None', color='k')
    if args.energy == 'MC':
        plt.xlabel('$\log_{10}(E_{\mathrm{MC}}/\mathrm{GeV})$')
    if args.energy == 'reco':
        plt.xlabel('$\log_{10}(E_{\mathrm{reco}}/\mathrm{GeV})$')
    ax.set_ylabel('Fraction correctly identified')
    ax.set_ylim([0.0, 1.0])
    ax.set_xlim([6.2, 9.5])
    # ax.set_xscale('log', nonposx='clip')
    plt.legend()
    if args.energy == 'MC':
        outfile = args.outdir + '/fraction-reco-correct_vs_MC-energy.png'
    if args.energy == 'reco':
        outfile = args.outdir + '/fraction-reco-correct_vs_reco-energy.png'
    plt.savefig(outfile)
