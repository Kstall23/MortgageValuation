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
    data = data[data['HousingExpenseToIncome'] != 999]
    data = data[data['TotalDebtToIncome'] != 999]
    end = len(data)
    if start - end == 0:
        print(".......No missing values, no rows dropped\n")
    else:
        print("......." + str(start-end) + " rows dropped with at least one missing value\n")
    return data

def removeExtremeOutliers(data, columns):
    print("....Removing extreme outliers....")
    print(".......Dropping rows with values outside the whiskers of length 3 x IQR\n")
    
    # set up columns to check for outliers
    columnsForOutliers = columns.copy()
    safeCols = ['Year', 'B1CreditScore', 'B2CreditScore']
    for col in safeCols:
        columnsForOutliers.remove(col)

    # gather quartile limits for each column
    limits = {}
    for col in columnsForOutliers:
        limits[col] = (3 * (data[col].quantile(.75) - data[col].quantile(.25))) + data[col].quantile(.75)

    # remove outliers from each column
    start = len(data)
    thisStart = len(data)
    for col in columnsForOutliers:
        data = data[data[col] < limits[col]]
        
        end = len(data)
        print("+++ Outliers removed from " + col + " column: " + str(thisStart - end))
        thisStart = end

    print("\n" + str(start-end) + " rows with outliers dropped from the dataset\n")

    return data


def normalize(data, columns):
    print("....Standardizing the columns of data....")
    print(".......Setting each column's values to have mean of 0 and std of 1\n")
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

    return standDataFrame, columns


def create_folds(data):
    print("....Drawing out a 10% random sample....")
    print(".......10% set aside for test data, remaining 90% for training")
    train, test = train_test_split(data, test_size=0.1)
    print(".........." + str(len(train)) + " points in the training set and " + str(len(test)) + " points in the test set\n")
    return train, test