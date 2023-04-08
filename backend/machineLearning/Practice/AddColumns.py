import numpy as np
import pandas as pd
from datetime import datetime
import git
from git import Repo
import os

# ---------------------------------------------------------------
# Check that we are in the MortageValuation directory
def checkDirectory():
    current_dir = os.getcwd()
    parts = current_dir.split("\\")
    if parts[-1] != "MortgageValuation":
        exit("ERROR ::::: Code is not running in the MortgageValuation repository directory!!")
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Check for a conneciton to the dataset repository, should be in same folder as this repo, but not in this repo
# ==CREATE== a connection if there isn't one
# ===PULL=== current remote repository status 
def getRepo():
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

    repo.remotes.origin.pull()       # pull most recent repo down locally

    return repo, repo_dir
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# ==LOAD== some data into a DataFrame
def readData(repo_dir, input_file_name, columns):
    data = pd.read_csv(os.path.join(repo_dir, input_file_name), sep=',', names=columns, header=0)
    print("...Reading an input file from remote repo...")
    return data
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# ==PUSH== new changes back to the remote repo, 
# added a bunch of prints to show repo status throughout
def pushRepo(output_file_name):
    print("\n...file edited")
    print(repo.git.status())

    repo.index.add([output_file_name])
    print("\n...file added")
    print(repo.git.status())

    repo.index.commit("Some data")
    print("\n...file committed")
    print(repo.git.status())

    repo.remotes.origin.push()
    print("\n...file pushed")
    print(repo.git.status())
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# Run the functions

checkDirectory()                # check if we're working in right folder
repo, repo_dir = getRepo()      # create or load the repo, pull current status

# Get ready to load data
input_file_name = "Mortgage Dataset.csv"
columns = ['Year', 'MonthlyIncome', 'UPBatAcquisition', 'LTVRatio', 'BorrowerCount', 'InterestRate', 'OriginationValue', 'HousingExpenseToIncome', 'TotalDebtToIncome', 'B1CreditScore', 'B2CreditScore']
data = readData(repo_dir, input_file_name, columns)    # read data into pandas DataFrame


''' # Now edit the dataframe and push it to a new Excel File '''
newData = data.copy()

# Add column for Loan Performance
newData['Performance'] = ["Current"] * len(data)

# Add column for original property value
valueColumn = []
for index, row in data.iterrows():
    valueColumn.append(row['OriginationValue'] / (row['LTVRatio'] / 100))
newData['PropertyValue'] = valueColumn

# Add column for current property value
newData['CurrentPropertyValue'] = valueColumn

# Add column for the change in property value
changes = []
for index, row in newData.iterrows():
    changes.append((row['PropertyValue'] - row['CurrentPropertyValue']) / row['PropertyValue'])
newData['ValueChange'] = changes

# Write out this new dataframe to a csv
output_file_name = "NewData.csv"
newData.to_csv(os.path.join(repo_dir, output_file_name))



pushRepo(output_file_name)        # push local changes to the remote repository
