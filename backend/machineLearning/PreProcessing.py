import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# VARIABLES - found in previous exploratory analysis
# NUM_CLUSTERS = 8
NUM_CLUSTERS = 30
NUM_PCS = 6

''' ============================================================= '''
''' === Functions for handling some preprocessing of the data === '''
''' ============================================================= '''

def removeMissing(data):
    print("\n....Dropping rows with Missing Values....")
    start = len(data)
    data = data.dropna()
    data = data[data['HousingExpenseToIncome'] != 999]
    data = data[data['TotalDebtToIncome'] != 999]
    end = len(data)
    if start - end == 0:
        print(".......No missing values, no rows dropped")
    else:
        print("......." + str(start-end) + " rows dropped with at least one missing value")
    return data


def removeExtremeOutliers(data, columns):
    print("\n....Removing extreme outliers....")
    print(".......Dropping rows with values outside the whiskers of length 3 x IQR\n")
    
    # gather quartile limits for each column
    limits = {}
    for col in columns:
        limits[col] = (3 * (data[col].quantile(.75) - data[col].quantile(.25))) + data[col].quantile(.75)

    # remove outliers from each column
    start = len(data)
    thisStart = len(data)
    for col in columns:
        data = data[data[col] < limits[col]]
        
        end = len(data)
        print("+++ Outliers removed from " + col + " column: " + str(thisStart - end))
        thisStart = end

    print("\n" + str(start-end) + " rows with outliers dropped from the dataset")
    print("...{} rows remaining...".format(end))

    return data


def normalize(data, columns):
    print("\n....Standardizing the columns of data....")
    print(".......Setting each column's values to have mean of 0 and std of 1")
    years = data['Year']
    data = data.drop(columns=['Year'])
    columns.remove('Year')

    ss = preprocessing.StandardScaler()
    standData = ss.fit_transform(data)

    # reshape years to concatenate to data
    # years = np.asarray(years).reshape(len(years), 1)
    # standData = np.concatenate((years, standData), axis=1)
    # columns.insert(0, 'Year')

    # transform back into a DataFrame so we can still use column headers
    standDataFrame = pd.DataFrame(standData, columns=columns)

    return standDataFrame, columns, ss


def create_folds(data):
    print("....Drawing out a 10% random sample....")
    print(".......10% set aside for test data, remaining 90% for training")
    train, test = train_test_split(data, test_size=0.1)
    print(".........." + str(len(train)) + " points in the training set and " + str(len(test)) + " points in the test set\n")
    return train, test


''' ======================================================================== '''
''' === Functions for handling some preprocessing of the test data point === '''
''' ======================================================================== '''

def testHandleMissing(originalPoint):
    # handle missing MonthlyIncome
    # TODO: give it a value of 0

    # handle missing UPBatAcquisition
    # TODO: I don't think we can operate without this info, throw error?

    # handle missing LTVRatio
    # TODO: I don't think we can operate without this info, throw error?
    
    # handle missing BorrowerCount
    # TODO: give it a value of 1, B1 and B2 value of 9

    # handle missing InterestRate
    # TODO: I don't think we can operate without this info, throw error?

    # handle missing OriginationValue
    # TODO: I don't think we can operate without this info, throw error?

    # handle missing HousingExpenseToIncome
    # TODO: give it a value of 999

    # handle missing TotalDebtToIncome
    # TODO: give it a value of 999

    # handle missing B1CreditScore
    # TODO: give it a value of 9

    # handle missing B2CreditScore
    # TODO: give it a value of 9
    newPoint = originalPoint
    return newPoint