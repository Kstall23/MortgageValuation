import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle

import PreProcessing as pp
import GitFunctions as gf

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# VARIABLES - found in previous exploratory analysis
from PreProcessing import NUM_CLUSTERS
from PreProcessing import NUM_PCS

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
    fullColumns = ['Year', 'MonthlyIncome', 'UPBatAcquisition', 'LTVRatio', 'BorrowerCount', 'InterestRate', 'OriginationValue', 'HousingExpenseToIncome', 'TotalDebtToIncome', 'B1CreditScore', 'B2CreditScore', 'Performance', 'PropertyValue', 'CurrentPropertyValue', 'ValueChange', 'Price']
    fullData = gf.readData(repo_dir, input_file_name, fullColumns)    # read data into pandas DataFrame

    # Run some preprocessing methods on this data for clustering
    no_missing_data = pp.removeMissing(fullData)

    columnsForOutliers = ['MonthlyIncome', 'UPBatAcquisition', 'LTVRatio', 'BorrowerCount', 'InterestRate', 'OriginationValue', 'HousingExpenseToIncome', 'TotalDebtToIncome']
    no_outliers_data = pp.removeExtremeOutliers(no_missing_data, columnsForOutliers)

    columnsForClustering = ['Year', 'MonthlyIncome', 'UPBatAcquisition', 'LTVRatio', 'BorrowerCount', 'InterestRate', 'OriginationValue', 'HousingExpenseToIncome', 'TotalDebtToIncome', 'B1CreditScore', 'B2CreditScore']
    reduced_data = no_outliers_data[columnsForClustering]

    std_data, columns, ss = pp.normalize(reduced_data, columnsForClustering)
    # trainData, testData = pp.create_folds(std_data)

    return no_outliers_data, std_data, columns, ss, repo, repo_dir

# ------------------------------------------------

def getClusters(data):
    # Reduce Dimensionality with PCA
    pca = PCA(n_components = NUM_PCS)
    pca_data = pca.fit_transform(data)

    # Use KMeans to cluster the PCA-ed data
    kmeans = KMeans(n_clusters = NUM_CLUSTERS, n_init = 10)
    clusters = kmeans.fit_predict(pca_data)

    return kmeans.cluster_centers_, clusters, pca, kmeans

# ------------------------------------------------

def storeClusterData(repo_dir, std_centroids, columns, cluster_labels, fullData, ss, pca, kmeans):
    # all file names for the three database files and data manipulation objects
    file_names = ["StandardizedCentroids.csv", "ReadableCentroids.csv", "ClusterLabels.csv", "ss.pkl", "pca.pkl", "kmeans.pkl"]
    
    ''' ======================================================================================
        Turns out that storing these three files to the database is pointless, they won't be used in prediciton, 
        just good for human visualization of the process
        ====================================================================================== '''
    # Throw standardized centroids into a DataFrame and write it to a csv
    std_cent_df = pd.DataFrame(std_centroids, columns=["PC"+str(x) for x in range(1,NUM_PCS+1)])        # create a df
    std_cent_df.to_csv(os.path.join(repo_dir, "ModelOutputFiles", file_names[0]))           # write to csv in database repo directory

    # Translate standardized centroids back to original values that actually mean something, throw into a DataFrame
    read_cent_df = pd.DataFrame(ss.inverse_transform(pca.inverse_transform(std_centroids)), columns=columns)    # create a df
    read_cent_df.to_csv(os.path.join(repo_dir, "ModelOutputFiles", file_names[1]))          # write to csv in database repo directory

    # Store cluster labels for every point in the dataset as a csv
    np.savetxt(os.path.join(repo_dir, "ModelOutputFiles", file_names[2]), cluster_labels, delimiter=', ', fmt='% s', header="label")

    # ====================================================================================================== #

    ''' =======================================================================================
        These cluster files and the model files will make up our "database" and will be pulled by the prediction script
        ======================================================================================= '''
    # Store each data by cluster
    for i in range(NUM_CLUSTERS):
        mask = cluster_labels == i
        cluster_df = fullData[mask]
        cluster_file = str(i) + "ClusterData.csv"
        cluster_df.to_csv(os.path.join(repo_dir, "ModelOutputFiles", cluster_file))

    # Store StandardScaler and PCA objects 
    pickle.dump(ss, open(os.path.join(repo_dir, "ModelOutputFiles", file_names[3]), 'wb'))
    pickle.dump(pca, open(os.path.join(repo_dir, "ModelOutputFiles", file_names[4]), 'wb'))
    pickle.dump(kmeans, open(os.path.join(repo_dir, "ModelOutputFiles", file_names[5]), 'wb'))


    return file_names

# ------------------------------------------------

def trainingClustersDriver():

    print("Hey look, this loads data and decides how to cluster it.")

    # set up repo, load in the data, run preprocessing
    no_outliers_data, data, columns, ss, repo, repo_dir = getData()

    # Cluster the training data
    std_centroids, cluster_labels, pca, kmeans = getClusters(data)

    # Write cluster centroids to csv's in standardized and readable form
    # Also write the cluster labels to a file
    # Also save the data manipulation objects to a file for later use on test data (StandardScaler and PCA objects)
    file_names = storeClusterData(repo_dir, std_centroids, columns, cluster_labels, no_outliers_data, ss, pca, kmeans)

    # Push these files to the 'database'
    msg = datetime.now().strftime("%d-%m-%y %H:%M") + " Cluster Database Update"
    gf.pushRepo(repo, 'ModelOutputFiles/', msg)
    
# ------------------------------------------

trainingClustersDriver()
