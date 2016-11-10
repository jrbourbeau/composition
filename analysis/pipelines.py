#!/usr/bin/env python

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Use get_pipeline to ensure that same hyperparameters are used each time a
# classifier is needed, and that the proper scaling is always done before
# fitting
def get_pipeline(classifier_name):
    ''' Returns classifier pipeline '''

    if classifier_name == 'RF':
        classifier = RandomForestClassifier(
            n_estimators=100, max_depth=8, n_jobs=10, random_state=2)
    elif classifier_name == 'KN':
        classifier = KNeighborsClassifier(n_neighbors=200, n_jobs=10)
    else:
        raise('{} is not a valid classifier name...'.format(classifier_name))

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)])

    return pipeline
