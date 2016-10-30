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
    p.add_argument('--outdir', dest='outdir',
                   default='/home/jbourbeau/public_html/figures/composition',
                   help='Output directory')
    args = p.parse_args()

    '''Throughout this code, X will represent features,
       while y will represent class labels'''

    df = load_sim()
    # Preprocess training data
    # feature_list = np.array(['reco_log_energy', 'reco_cos_zenith', 'InIce_log_charge',
    feature_list = np.array(['reco_log_energy', 'ShowerPlane_cos_zenith', 'InIce_log_charge',
                             'NChannels', 'NStations', 'reco_radius', 'reco_InIce_containment', 'log_s125'])
    num_features = len(feature_list)
    X, y = df[feature_list].values, df.MC_comp.values
    # Convert comp string labels to numerical labels
    le = LabelEncoder().fit(y)
    y  = le.transform(y)

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
    forest.fit(X_train_std, y_train)
    name = forest.__class__.__name__
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots()
    feature_labels = np.array(['$\\log_{10}({\mathrm{E/GeV})$', '$\cos(\\theta)$', 'InIce charge',
                               'NChannels', 'NStations', 'IT reco radius', 'InIce containment', 'log(s125)'])
    for f in range(num_features):
        print('{}) {}'.format(f + 1, importances[indices[f]]))

    plt.ylabel('Feature Importances')
    plt.bar(range(num_features),
            importances[indices],
            align='center')

    plt.xticks(range(num_features),
               feature_labels[indices], rotation=90)
    plt.xlim([-1, len(feature_list)])
    outfile = args.outdir + '/random_forest_feature_importance.png'
    plt.savefig(outfile)

    d = pd.DataFrame(df, columns=feature_list)
    # Compute the correlation matrix
    corr = d.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots()
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0,
                square=True, xticklabels=feature_labels, yticklabels=feature_labels,
                linewidths=.5, cbar_kws={'label': 'Covariance'}, annot=True, ax=ax)
    outfile = args.outdir + '/feature_covariance.png'
    plt.savefig(outfile)

    print('=' * 30)
    print(name)
    test_predictions = forest.predict(X_test_std)
    test_acc = accuracy_score(y_test, test_predictions)
    print('Test accuracy: {:.4%}'.format(test_acc))
    train_predictions = forest.predict(X_train_std)
    train_acc = accuracy_score(y_train, train_predictions)
    print('Train accuracy: {:.4%}'.format(train_acc))
    print('=' * 30)

    # # Plotting decision regions
    # fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
    # # plot_decision_regions(X_train_std, y_train, clf=forest,
    # #                       res=0.02, legend=2)
    # # ax.scatter(X_train[:, 0][correctly_identified_mask & (reco_comp == 'P')],
    # #            X_train[:, 2][correctly_identified_mask & (reco_comp == 'P')],
    # #            label='P', color='b', alpha=)
    # # ax.scatter(X_train[:, 0][correctly_identified_mask & (reco_comp == 'Fe')],
    # #            X_train[:, 2][correctly_identified_mask & (reco_comp == 'Fe')],
    # #            label='Fe', color='g')
    # sns.jointplot(X_train[:, 0][(reco_comp == 'P')],
    #            X_train[:, 2][(reco_comp == 'P')],
    #            label='P', kind='scatter', color='b', ax=axarr[0,0])
    # sns.joinplot(X_train[:, 0][(reco_comp == 'Fe')],
    #            X_train[:, 2][(reco_comp == 'Fe')],
    #            label='Fe', kind='scatter', color='g', ax=axarr[0,1])
    # sns.jointplot(X_train[:, 0][(reco_comp == 'P')],
    #            X_train[:, 2][(reco_comp == 'P')],
    #            label='P', kind='scatter', color='b', ax=axarr[1,0])
    # sns.jointplot(X_train[:, 0][(reco_comp == 'Fe')],
    #            X_train[:, 2][(reco_comp == 'Fe')],
    #            label='Fe', kind='scatter', color='g', ax=axarr[1,0])
    # # Adding axes annotations
    # plt.xlabel('Energy/GeV')
    # plt.ylabel('InIce Charge')
    # plt.legend()
    # # plt.title('SVM on Iris')
    # outfile = args.outdir + '/decision_regions.png'
    # plt.savefig(outfile)
