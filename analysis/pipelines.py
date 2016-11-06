#!/usr/bin/env python

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def RF_pipeline():
    ''' Returns random forest classifier pipeline '''

    forest = RandomForestClassifier(
        n_estimators=100, max_depth=6, criterion='gini', max_features=None, n_jobs=-1)

    pipeline = Pipeline([
        ('stdsc', StandardScaler()),
        ('RF', forest)])

    return pipeline


def KNeighbors_pipeline():
    ''' Returns KNeighbors classifier pipeline '''

    kneighbors = KNeighborsClassifier()

    pipeline = Pipeline([
        ('stdsc', StandardScaler()),
        ('KNeighbors', kneighbors)])

    return pipeline

# Use get_pipeline to ensure that same hyperparameters are used each time a
# classifier is needed, and that the proper scaling is always done before
# fitting
def get_pipeline(classifier_name):
    ''' Returns classifier pipeline '''

    if classifier_name == 'RF':
        classifier = RandomForestClassifier(
            n_estimators=100, max_depth=6, criterion='gini', n_jobs=4)
    elif classifier_name == 'KN':
        classifier = KNeighborsClassifier(n_neighbors=80, n_jobs=4)
    else:
        raise('{} is not a valid classifier name...'.format(classifier_name))

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)])

    return pipeline
