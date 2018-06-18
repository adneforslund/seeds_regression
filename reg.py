# -*- encoding: utf-8 -*-

import numpy as np
import pandas as pd
import csv
from sklearn.ensemble import RandomForestRegressor


PATH_TO_TRAINING_SET = 'Flaveria.csv'


# change this string to the location of the test set
PATH_TO_TEST_SET = ''


def regression(train_file, test_file):
    #train
    train_set = pd.read_csv(train_file)
    train_set = train_set.replace({'N level': {'L': 1, 'M': 2, 'H': 3}})
    train_set = pd.get_dummies(train_set)

    w_train = train_set['Plant Weight(g)']
    matrix_train = train_set.iloc[:, np.r_[0, 2, 3, 4, 5, 6, 7]]
    

    #test
    test_set = pd.read_csv(test_file)
    test_set = test_set.replace({'N level': {'L': 1, 'M': 2, 'H': 3}})
    test_set = pd.get_dummies(train_set)

    w_test = train_set['Plant Weight(g)']
    matrix_test = test_set.iloc[:, np.r_[0, 2, 3, 4, 5, 6, 7]]

    #fitting
    regr = RandomForestRegressor(max_depth=5, n_estimators=200)
    regr.fit(matrix_train, w_train)

    print(" RandomForestRegressor: score: "   + str(regr.score(matrix_test, w_test)))


regression(PATH_TO_TRAINING_SET ,PATH_TO_TEST_SET)
