import numpy as np
import pandas as pd
from datetime import datetime
import os

import PreProcessing as pp
import GitFunctions as gf

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle

# VARIABLES - found in previous exploratory analysis
from PreProcessing import NUM_CLUSTERS
from PreProcessing import NUM_PCS

# ------------------------------------------------

def getClusterData():
    # Check that we are in the MortageValuation directory
    gf.checkDirectory()
    
    # Check for a conneciton to the dataset repository, should be in same folder as this repo, but not in this repo
    # ==CREATE== a connection if there isn't one
    # ===PULL=== current repo files/status
    repo, repo_dir = gf.getRepo()

    ''' No longer need the cluster centroids - now we pull the kmeans model from the training script and use it for clustering the new point'''
    # ==LOAD== the cluster centroids into a DataFrame
    # input_file_name = "ModelOutputFiles/StandardizedCentroids.csv"
    # centroids = gf.readData(repo_dir, input_file_name, columns=["PC"+str(x) for x in range(1,NUM_PCS+1)])    # read data into pandas DataFrame

    ''' No longer need cluster labels - now we pull all the data for one cluster in a later step '''
    # ==LOAD== the cluster labels into a DataFrame
    # input_file_name = "ModelOutputFiles/ClusterLabels.csv"
    # cluster_labels = list(gf.readData(repo_dir, input_file_name, columns=['Label'])['Label'])   # read labels into a dataframe, convert to list

    # ==LOAD== the data manipulation objects for manipulating test data points
    object_files = ['ss.pkl', 'pca.pkl', 'kmeans.pkl']
    ss = pickle.load(open(os.path.join(repo_dir, "ModelOutputFiles", object_files[0]), 'rb'))
    pca = pickle.load(open(os.path.join(repo_dir, "ModelOutputFiles", object_files[1]), 'rb'))
    kmeans = pickle.load(open(os.path.join(repo_dir, "ModelOutputFiles", object_files[2]), 'rb'))


    return repo, repo_dir, ss, pca, kmeans

# ------------------------------------------------

def getTestPoint(repo_dir):

    '''
    =================================================
    == THIS NEEDS TO CHANGE 
    == Need to figure out how to pull the data point of the page we're viewing in the Mortgage Details
    == This currently pulls a dummy file from the database repo with one point stored in it
    =================================================
    '''
    # ==LOAD== the test point from somewhere
    input_file_name = "TestPoint1.csv"
    fullColumns = ['Year', 'MonthlyIncome', 'UPBatAcquisition', 'LTVRatio', 'BorrowerCount', 'InterestRate', 'OriginationValue', 'HousingExpenseToIncome', 'TotalDebtToIncome', 'B1CreditScore', 'B2CreditScore', 'Performance', 'PropertyValue', 'CurrentPropertyValue', 'ValueChange', 'Price']
    fullPoint = gf.readData(repo_dir, input_file_name, fullColumns)     # load the data point into a dataframe

    print(fullPoint['UPBatAcquisition'].iloc[0])

    return fullPoint

# ------------------------------------------------

def testPointProcessing(fullPoint, ss, pca):
    # Run some preprocessing methods on this data to reduce based on training data
    columns = ['MonthlyIncome', 'UPBatAcquisition', 'LTVRatio', 'BorrowerCount', 'InterestRate', 'OriginationValue', 'HousingExpenseToIncome', 'TotalDebtToIncome', 'B1CreditScore', 'B2CreditScore']
    reduced_point = fullPoint[columns]                      # reduce to clustering-relevant columns
    no_missing_point = pp.testHandleMissing(reduced_point)             # handle missing values by adding dummy values or throwing an error if it's in a crucial field

    # normalize the data based on training data StandardScaler
    std_point = pd.DataFrame(ss.transform(no_missing_point), columns=columns)

    # perform PCA on the standardized point to reduce to NUM_PCS features
    pca_point = pca.transform(std_point)
    
    return pca_point

# ------------------------------------------------

def provideSuggestion(point, repo_dir, kmeans, fullPoint):
    # Determine which cluster the new point belongs to
    [cluster] = kmeans.predict(point)

    # ==LOAD== the data associated with that cluster from the database repo
    cluster_file_name = "ModelOutputFiles/" + str(cluster) + "ClusterData.csv"
    fullColumns = ['Year', 'MonthlyIncome', 'UPBatAcquisition', 'LTVRatio', 'BorrowerCount', 'InterestRate', 'OriginationValue', 'HousingExpenseToIncome', 'TotalDebtToIncome', 'B1CreditScore', 'B2CreditScore', 'Performance', 'PropertyValue', 'CurrentPropertyValue', 'ValueChange', 'Price']
    fullClusterData = gf.readData(repo_dir, cluster_file_name, fullColumns)     # load the data point into a dataframe

    # Determine a price suggestion based on average price of cluster-mates
    # Every point in this cluster? Some fraction of the points?
    price = fullClusterData['Price'].mean()

    # Add flags if the loan is currently Delinquent or has significant (app/dep)reciation
    fullPoint = pd.Series(fullPoint.iloc[0])
    
    if fullPoint['Performance'] == "Current":
        delinq = False
    elif fullPoint['Performance'] == "Delinquent":
        delinq = True

    if fullPoint['ValueChange'] >= 25:            # flagging if value has appreciated by at least 25%
        appr = True
        depr = False
    elif fullPoint['ValueChange'] <= -25:         # flagging if value has depreciated by at least 25%
        depr = True
        appr = False
    else:
        appr = False
        depr = False

    return price, delinq, appr, depr

# ------------------------------------------------

def testOnePointDriver():

    print("Hey look, maybe someday this will provide a prediction point.")

    # set up repo, load in the cluster data
    repo, repo_dir, ss, pca, kmeans = getClusterData()

    # load in new test point
    fullPoint = getTestPoint(repo_dir)

    # preprocess the test point using data manipulation objects from training data
    point = testPointProcessing(fullPoint, ss, pca)

    # provide suggestion - place point in a cluster and draw pricing data from cluster members
    suggestionNumber, delinq, appr, depr = provideSuggestion(point, repo_dir, kmeans, fullPoint)

    print(suggestionNumber)
    print(delinq, appr, depr)

# ------------------------------------------------

testOnePointDriver()