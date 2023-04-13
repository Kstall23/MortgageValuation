import time, os, sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
backend_dir = os.path.join(parent_dir, "backend")
sys.path.append(backend_dir)
from machineLearning import predictionModel as pm

def predict(file_name):
    print("Running prediction on ", file_name, "...")
    return pm.testOnePointDriver()