import numpy as np
import pandas as pd
from datetime import datetime
import git
from git import Repo
import os

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
input_file_name = "TrainingData.csv"
data = pd.read_csv(os.path.join(repo_dir, input_file_name), sep=',', names=["Title", "Owner", "Asset Type", "Unpaid Principal Balance", "Loan Term", "Note Rate", "PPP", "Value", "LTV", "PropType", "Occupancy"], header=0)

print("...Reading an input file from remote repo...")
print(data["Owner"])


# create or edit a file
output_file_name = os.path.join(repo_dir, "output.txt")
out = open(output_file_name, "a")
# Loop through the DataFrame and write some data to output file
for index, row in data.iterrows():
    out.write("Row " + str(index) + "\n")
    out.write("Owner " + row['Owner'] + " has a " + row['Asset Type'] + " on this property: " + row['Title'])
    out.write("\n\n")
out.close()


# ==PUSH== new changes back to the remote repo, added a bunch of prints to show repo status throughout
time = str(datetime.now())
print("\n...file edited")
print(repo.git.status())

repo.index.add([output_file_name])
print("\n...file added")
print(repo.git.status())

repo.index.commit(time + " new output file commit")
print("\n...file committed")
print(repo.git.status())

repo.remotes.origin.push()
print("\n...file pushed")
print(repo.git.status())
