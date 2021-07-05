#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Loading training data and extracting features

print(__doc__)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import pandas as pd

import numpy as np

def train_test_svc_ftd(total_energy, time, trainingdata_size, data):
    """
    Train and test SVC model using fixed training data set for either 2 (x,px) 
    or 3 (x,px,LD) features
    """
    
    print(data.shape,'\n',data.head)

    data = data.dropna()
    print('Samples on the energy surface:%d'%(data.shape[0]),'\n',data.head)


    if data.shape[1] == 5:
        X = data.drop(["y", "p_y", "Exit channel"], axis=1)
        # X = data.drop(["Exit channel"], axis=1)
        y = data["Exit channel"].astype(int)
    elif data.shape[1] == 7:
        X = data.drop(["y", "p_y", "TE", "Exit channel"], axis=1)
        # X = data.drop(["Exit channel"], axis=1)
        y = data["Exit channel"].astype(int)
        

    #scaler = StandardScaler()
    #X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                        test_size=0.2, random_state=0)



    #%% Kernel parameter optimization using grid search with cross-validation

    # Set the parameters by cross-validation
    # C_range = np.logspace(-3, 10, 14)
    # gamma_range = np.logspace(-5, 5, 11)
    # C_range  = [1, 10, 1e2]
    # gamma_range = [1e-1, 1, 10]
    # C_range  = [10, 100, 1000]
    # gamma_range = [1, 10, 100]
    C_range  = [100, 1000, 1e4, 1e5]
    gamma_range = [10, 100, 1e3, 1e4]
    tuned_parameters = [{'kernel': ['rbf'], \
                         'gamma': gamma_range, \
                         'C': C_range}]

    #tuned_parameters = [{'kernel': ['rbf'], \
    #                     'gamma': [1e2, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4], \
    #                     'C': [1, 10, 1e2, 1e3, 1e4, 1e5]}]
    #scoring_metric = 'accuracy'
    scoring_metric = 'f1_weighted'

    clf = GridSearchCV(SVC(), tuned_parameters, scoring = scoring_metric, cv = 5)
    #clf = GridSearchCV(SVC(), tuned_parameters, scoring = 'f1_weighted', cv = 5)

    clf.fit(X_train, y_train)

    print("The best parameters are %s with a score of %0.2f"
          % (clf.best_params_, clf.best_score_))

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()

    y_true, y_pred = y_test, clf.predict(X_test)
    report = classification_report(y_true, y_pred, output_dict=True)
    print(classification_report(y_true, y_pred, output_dict=True))
    print()

    df = pd.DataFrame(report).transpose()
    df.to_csv('hh_classificationreport_samples%d'%(trainingdata_size) \
              + '_E%.3f'%(total_energy) \
              + '_T%.3f'%(time) + '.csv') 

    # save the model to disk
    joblib.dump(clf, 'hh_svc_samples%d'%(trainingdata_size) \
                + '_E%.3f'%(total_energy) \
                + '_T%.3f'%(time) + '.sav')


    return 


if __name__ == '__main__':
    train_test_svc_ftd()


    
    
#%%

# svclassifier = SVC(kernel="linear", C = 10)
# svclassifier.fit(X_train, y_train)

# polynomial
# svclassifier = SVC(kernel="poly", degree=8)
# svclassifier.fit(X_train, y_train)

# gaussian
# svclassifier = SVC(kernel='rbf', C = 4000, gamma = 50)
# svclassifier.fit(X_train, y_train)
# score = svclassifier.score(X_test, y_test)

# y_pred = svclassifier.predict(X_test)

# from sklearn.metrics import classification_report, confusion_matrix

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(score)





