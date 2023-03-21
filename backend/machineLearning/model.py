import numpy as np
import pandas as pd
import os
os.chdir('..')

print("Hey look, maybe someday this will provide a prediction point.")

# read in training data file as a pandas dataframe, assigning appropriate headers
data = pd.read_csv('MortgageValuation\\backend\database\TrainingData.csv', sep=',', names=["Title", "Owner", "Asset Type", "Unpaid Principal Balance", "Loan Term", "Note Rate", "PPP", "Value", "LTV", "PropType", "Occupancy"])

for index, row in data.iterrows():

    print("Row "+ str(index))
    for val in row:
        print(val)
    print()
    print()