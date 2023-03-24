import numpy as np
import pandas as pd
from datetime import datetime
import git
from git import Repo
import os



os.chdir('..')
repo_dir = os.path.join(os.getcwd(), "Brightvine_model_files")
input_file_name = "Mortgage Dataset.csv"
data = pd.read_csv(os.path.join(repo_dir, input_file_name), sep=',', names=['Year', 'MonthlyIncome', 'UPBatAcquisition', 'LTVRatio', 'BorrowerCount', 'InterestRate', 'OriginationValue', 'HousingExpenseToIncome', 'TotalDebtToIncome', 'B1CreditScore', 'B2CreditScore'], header=0)


# Summary Statistics
print(data.describe())
nines = data[data['HousingExpenseToIncome'] == 999]
print(nines)
print(len(nines) / len(data))

print()
print()

# Correlation Matrix
# corr = data.corr()
# print(corr)
# print()