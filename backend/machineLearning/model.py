import numpy as np
import pandas as pd
from datetime import datetime
import git
from git import Repo
import os
import PreProcessing

def main():

    print("Hey look, maybe someday this will provide a prediction point.")


    # Check that we are in the MortageValuation directory
    current_dir = os.getcwd()
    parts = current_dir.split("\\")
    if parts[-1] != "MortgageValuation":
        exit("ERROR ::::: Code is not running in the MortgageValuation repository directory!!")

    # Check for a conneciton to the dataset repository, should be in same folder as this repo, but not in this repo
    # ==CREATE== a connection if there isn't one
    os.chdir('..')
    repo_dir = os.getcwd() + "\\Brightvine_model_files"         # path of data repo
    try:
        repo = Repo(repo_dir)       # works if it exists
    except:                         # if error, enters this chunk to initialize it
        print("Repo does not yet exist locally")
        print(".....Creating and Cloning now.....")
        # initiate new Git repo
        git.Git(repo_dir).clone("https://github.com/jjbrown23/Brightvine_model_files.git")
        repo = Repo(repo_dir)
    finally:
        assert not repo.bare


    # ==PULL== some data and read it
    repo.remotes.origin.pull()       # pull most recent repo down locally
    input_file_name = "Mortgage Dataset.csv"
    data = pd.read_csv(os.path.join(repo_dir, input_file_name), sep=',', names=['Year', 'MonthlyIncome', 'UPBatAcquisition', 'LTVRatio', 'BorrowerCount', 'InterestRate', 'OriginationValue', 'HousingExpenseToIncome', 'TotalDebtToIncome', 'B1CreditScore', 'B2CreditScore'], header=0)

    print("...Reading an input file from remote repo...")
    print(data.head())
    print(data["LTVRatio"][:10])
    print(len(data))

    # Run some preprocessing methods on this data
    data = PreProcessing.removeMissing(data)
    data = PreProcessing.normalize(data)
    trainData, testData = PreProcessing.create_folds(data)
    


    # Try some K - Means Clustering?

    from sklearn.cluster import KMeans
    


    # Write the cluster centroid values back to a file and push to github?


# ------------------------------------------

main()
