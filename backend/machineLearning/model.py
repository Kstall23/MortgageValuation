import numpy as np
import pandas as pd
from datetime import datetime
import os
os.chdir('..')

import git
from git import Repo

print("Hey look, maybe someday this will provide a prediction point.")


# Figuring out GitPython

# initialize Repo object with existing repository
repo_dir = "C:\\Users\\Jacob\\OneDrive\\Documents\\Jacob\\4-2023 Spring\\Capstone\\Brightvine_model_files"
repo = Repo(repo_dir)
assert not repo.bare
print(repo.git.status())



# pull and read a file

repo.remotes.origin.pull()      # pull repo
# read data file into DataFrame
data = pd.read_csv(os.path.join(repo_dir, "TrainingData.csv"), sep=',', names=["Title", "Owner", "Asset Type", "Unpaid Principal Balance", "Loan Term", "Note Rate", "PPP", "Value", "LTV", "PropType", "Occupancy"], header=0)

# open output file, created if doesn't exist
file_name = os.path.join(repo_dir, "output.txt")
out = open(file_name, "a")

# Loop through DataFrame, write data to the output file
for index, row in data.iterrows():
    out.write("Row "+ str(index) + "\n")
    for val in row:
        out.write(val + "\n")
    out.write("\n\n")

# close the file
out.close()

# push the changes back to repo
time = str(datetime.now())
repo.index.add([file_name])
repo.index.commit(time + " new output file commit")
repo.remotes.origin.push()
print(repo.git.status())