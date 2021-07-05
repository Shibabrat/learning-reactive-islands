#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Support vector classifier using adaptive training data
"""

print(__doc__)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import pandas as pd

import numpy as np
from scipy.integrate import solve_ivp

import henonheiles
import importlib
importlib.reload(henonheiles)
import henonheiles as HH2dof

# y_constant = 0
# y_constant = -0.25
y_constant = 0.5

def event_return_sos(t, states, *params):
    if abs(t) < 1e-2:
        val = 10
    else:
        val = states[1] - y_constant
    return val
event_return_sos.terminal = False
event_return_sos.direction = 0


def get_extra_training_data(support_vectors, system_parameters, total_energy, end_time):
    """
    Generate training data using a uniform random distribution around each of the existing 
    support vectors
    """
    X = []
    y = []
    
    RelTol = 3.e-10
    AbsTol = 1.e-10 

#     sigma = np.sqrt(2.0*gamma)
    for idx, val in enumerate(support_vectors):
        center = val
        spread = np.array([[1, 0],[0, 1]])
        x_new, px_new = np.random.default_rng().multivariate_normal(center, spread, 10).T
        for i in range(len(x_new)):
            py_new = HH2dof.momentum_fixed_energy(x_new[i], y_constant, px_new[i], \
                                                  system_parameters, total_energy)
            if ~np.isnan(py_new):
                init_cond = np.array([x_new[i], y_constant, px_new[i], py_new])
                        
                sol = solve_ivp(HH2dof.vector_field, [0, end_time], init_cond, \
                                args = system_parameters, \
                                events = (HH2dof.event_escape_left, HH2dof.event_escape_right, \
                                          HH2dof.event_escape_top, event_return_sos), \
                                dense_output = True, rtol = RelTol, atol = AbsTol)

                if np.size(sol.t_events) > 0:
                    if np.size(sol.t_events[0]) == 1 and np.size(sol.t_events[3]) == 1:
                        escape_label = 1
                    elif np.size(sol.t_events[1]) == 1 and np.size(sol.t_events[3]) == 1:
                        escape_label = 2
                    elif np.size(sol.t_events[2]) == 1 and np.size(sol.t_events[3]) == 0:
                        escape_label = 3
                    else:
                        escape_label = 0
                else:
                    escape_label = 0
                
                X.append([x_new[i], px_new[i]])
                y.append(escape_label)
    
    return X, y



#%%

# m_x, m_y, omega_x, omega_y, delta
system_parameters = [1.0, 1.0, 1.0, 1.0, 1.0] # parameters from Demian-Wiggins (2017)
total_energy = 0.19
time = 30

# datapath = './hh2dof_firstRIs/adaptive_trainingdata_size/'
# datapath = './hh2dof_firstRIs_sos_y-0.25/adaptive_trainingdata_size/'
datapath = './hh2dof_firstRIs_sos_y0.5/adaptive_trainingdata_size/'

data = pd.read_csv(datapath + 'hh_escape_samples625_E%.3f'%(total_energy) \
                   + '_T%.3f'%(time) + '.txt', \
                   sep = " ", header = None, \
                   names = ["x", "y", "p_x", "p_y","Exit channel"])

#print(data.shape,'\n',data.head)

data = data.dropna()
print('Samples on the energy surface:%d'%(data.shape[0]),'\n',data.head)


X = data.drop(["y", "p_y", "Exit channel"], axis=1)
# X = data.drop(["Exit channel"], axis=1)
y = data["Exit channel"].astype(int)


X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                    test_size=0.2, random_state=0)


target_metric = 0
while target_metric < 0.995:

    C_range  = [10, 100, 1000]
    gamma_range = [1, 10, 100]
#     C_range  = [100]
#     gamma_range = [10]
    tuned_parameters = [{'kernel': ['rbf'], \
                         'gamma': gamma_range, \
                         'C': C_range}]
    scoring_metric = 'f1_weighted'
    
    #clf = GridSearchCV(SVC(), tuned_parameters, scoring = None, cv = 5)
    clf = GridSearchCV(SVC(), tuned_parameters, scoring = scoring_metric, cv = 5)
    
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
    
    support_vectors = clf.best_estimator_.support_vectors_
    print("Number of support vectors: %6d"%(len(support_vectors)))
    
    target_metric = clf.best_score_
    
    #increase training data by obtaining new support vectors
    X_train_new, y_train_new = get_extra_training_data(support_vectors, system_parameters, \
                                                       total_energy, time)

    X_train_new = pd.DataFrame(data = X_train_new, columns = ["x", "p_x"])
    X_train = X_train.append(X_train_new)
    
    y_train_new = pd.Series(y_train_new)
    y_train = y_train.append(y_train_new)
    
# save performance data and model 
df = pd.DataFrame(report).transpose()
df.to_csv('hh_classificationreport_samples625_E%.3f'%(total_energy) \
          + '_T%.3f'%(time) + '.csv')
joblib.dump(clf, 'hh_svc_samples625_E%.3f'%(total_energy) \
            + '_T%.3f'%(time) + '.sav')




