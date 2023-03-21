import numpy as np
import pandas as pd
import tensorflow
import sklearn.preprocessing as pre


print("Hey look, maybe someday this will provide a prediction point.")

# read in training data file as a pandas dataframe, assigning appropriate headers
# --- Preprocess Computer Hardware Data -----------------
hardware = pd.read_csv('backend\machineLearning\machine.data',sep=',',names=['vendor','model','myct','mmin','mmax','cach','chmin', 'chmax','predict','erp']) # predict is the value of interest

# drop these categorical columns that aren't useful
hardware.drop('vendor',inplace=True,axis=1)
hardware.drop('model',inplace=True,axis=1)

# Pull out prediction column
y = hardware['predict'].values
hardware.drop('predict',inplace=True,axis=1)
print(y)

# Set aside and normalize data attributes
sc = pre.StandardScaler()
X = hardware.iloc[:,0:7].values
# print(X)
X = sc.fit_transform(X)
# print(X)


# Build the model
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import MeanSquaredError

model = Sequential()
model.add(Dense(5, input_dim=7, activation="relu"))     # first hidden layer, 5 nodes
model.add(Dense(5, activation="relu"))                  # second hidden layer, 5 nodes
model.add(Dense(1, activation="linear"))                # output layer, 1 node

# compile the model
model.compile(loss="mse", optimizer="adam", metrics=[MeanSquaredError()])


# Train the model
model.fit(X, y, epochs=1500)

# Evaluate the model
_, accuracy = model.evaluate(X, y) 
print("Model MSE: " + str(accuracy))


# new_point = np.asarray([[7, 1, 5, 1, 4, 8, 2, 3, 5]]).astype(np.float32)
# prediction = model.predict(new_point)
# print([round(x[0]) for x in prediction])



