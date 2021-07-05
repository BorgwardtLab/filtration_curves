

import csv
import glob
import os
import argparse
import warnings 
import time
from tqdm import tqdm
import random
import json

import pandas as pd
from scipy.spatial import distance

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from utils import *

def run_rf(K, y, args, n_iterations=10):
    random.seed(42)

    iteration_metrics = create_metric_dict()
    iteration_accuracies = []
    for iteration in range(n_iterations):
        
        fold_metrics = create_metric_dict()
        fold_accuracies = []    
                
        cv = StratifiedKFold(
            n_splits=10,
            random_state=42 + iteration,
            shuffle=True
            ) 

        for train_index, test_index in cv.split(np.zeros(len(y)), y):
            D_train = [K[i] for i in train_index]
            y_train = [y[i] for i in train_index]
            
            D_test = [K[i] for i in test_index]
            y_test = [y[i] for i in test_index]

            clf = RandomForestClassifier(
                    max_depth=None, 
                    random_state=0,
                    n_estimators=1000, 
                    class_weight="balanced",
                    ) 

            clf.fit(D_train, y_train)
            y_pred = clf.predict(D_test)
            
            with warnings.catch_warnings(): 
                warnings.filterwarnings('ignore')

                fold_metrics = compute_fold_metrics(y_test, y_pred,
                        fold_metrics)
        iteration_metrics = update_iteration_metrics(fold_metrics,
                iteration_metrics)
    
    
    print_iteration_metrics(iteration_metrics)



