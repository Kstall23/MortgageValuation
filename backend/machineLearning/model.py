import numpy as np
import pandas as pd
from datetime import datetime
import os

import PreProcessing as pp
import GitFunctions as gf

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# VARIABLES - found in previous exploratory analysis
NUM_CLUSTERS = 8
NUM_PCS = 6

# ------------------------------------------------

def getData():
    # Check that we are in the MortageValuation directory
    gf.checkDirectory()
    
    # Check for a conneciton to the dataset repository, should be in same folder as this repo, but not in this repo
    # ==CREATE== a connection if there isn't one
    # ===PULL=== current repo files/status
    repo, repo_dir = gf.getRepo()

    # ==LOAD== some data into a DataFrame
    input_file_name = "NewData.csv"
    fullColumns = ['Year', 'MonthlyIncome', 'UPBatAcquisition', 'LTVRatio', 'BorrowerCount', 'InterestRate', 'OriginationValue', 'HousingExpenseToIncome', 'TotalDebtToIncome', 'B1CreditScore', 'B2CreditScore', 'Performance', 'PropertyValue', 'CurrentPropertyValue', 'ValueChange']
    fullData = gf.readData(repo_dir, input_file_name, fullColumns)    # read data into pandas DataFrame

    # Run some preprocessing methods on this data for clustering
    columns = ['Year', 'MonthlyIncome', 'UPBatAcquisition', 'LTVRatio', 'BorrowerCount', 'InterestRate', 'OriginationValue', 'HousingExpenseToIncome', 'TotalDebtToIncome', 'B1CreditScore', 'B2CreditScore']
    data = fullData[columns]
    data = pp.removeMissing(data)
    data = pp.removeExtremeOutliers(data, columns)
    data, columns, ss = pp.normalize(data, columns)
    trainData, testData = pp.create_folds(data)

    return data, columns, trainData, testData, ss, repo, repo_dir

# ------------------------------------------------

def getClusters(data, columns):
    # Reduce Dimensionality with PCA
    pca = PCA(n_components = NUM_PCS)
    pca_data = pca.fit_transform(data)

    # Use KMeans to cluster the PCA-ed data
    kmeans = KMeans(n_clusters = NUM_CLUSTERS, n_init = 10)
    clusters = kmeans.fit_predict(pca_data)

    return kmeans.cluster_centers_, clusters, pca

# ------------------------------------------------

def storeClusterData(std_centroids, cluster_labels, pca, ss, columns, repo_dir):
    file_names = ["StandardizedCentroids.csv", "ReadableCentroids.csv", "ClusterLabels.csv"]    # all file names for the three database files
    
    # Throw standardized centroids into a DataFrame and write it to a csv
    std_cent_df = pd.DataFrame(std_centroids, columns=["PC"+str(x) for x in range(1,NUM_PCS+1)])        # create a df
    std_cent_df.to_csv(os.path.join(repo_dir, "ModelOutputFiles", file_names[0]))           # write to csv in database repo directory

    # Translate standardized centroids back to original values that actually mean something, throw into a DataFrame
    read_cent_df = pd.DataFrame(ss.inverse_transform(pca.inverse_transform(std_centroids)), columns=columns)    # create a df
    read_cent_df.to_csv(os.path.join(repo_dir, "ModelOutputFiles", file_names[1]))          # write to csv in database repo directory

    # Store cluster labels for every point in the dataset as a csv
    np.savetxt(os.path.join(repo_dir, "ModelOutputFiles", file_names[2]), cluster_labels, delimiter=', ', fmt='% s')

    return file_names

# ------------------------------------------------

def trainingClustersDriver():

    print("Hey look, maybe someday this will provide a prediction point.")

    # set up repo, load in the data, run preprocessing
    data, columns, trainData, testData, ss, repo, repo_dir = getData()      # it's okay if most of these variables don't get used

    # Cluster the training data
    std_centroids, cluster_labels, pca = getClusters(data, columns)

    # Write cluster centroids to csv's in standardized and readable form
    # Also write the cluster labels to a file
    file_names = storeClusterData(std_centroids, cluster_labels, pca, ss, columns, repo_dir)

    # Push these files to the 'database'
    msg = str(datetime.now()) + " Cluster Database Update"
    gf.pushRepo(repo, 'ModelOutputFiles/', msg)
    
# ------------------------------------------

trainingClustersDriver()
