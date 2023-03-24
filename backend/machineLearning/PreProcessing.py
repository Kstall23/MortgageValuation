import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

''' ============================================================= '''
''' === Functions for handling some preprocessing of the data === '''
''' ============================================================= '''

def removeMissing(data):
    print("....Dropping rows with Missing Values....")
    start = len(data)
    data = data.dropna()
    end = len(data)
    if start - end == 0:
        print(".......No missing values, no rows dropped\n")
    else:
        print("......." + str(start-end) + " rows dropped with at least one missing value\n")
    return data


def normalize(data):
    print("....Standardizing the columns of data....")
    print(".......Setting each column's values to have mean of 0 and std of 1\n")
    ss = preprocessing.StandardScaler()
    standData = ss.fit_transform(data)
    return standData


def create_folds(data):
    print("....Drawing out a 10% random sample....")
    print(".......10% set aside for test data, remaining 90% for training")
    train, test = train_test_split(data, test_size=0.1)
    print(".........." + str(len(train)) + " points in the training set and " + str(len(test)) + " points in the test set\n")
    return train, test