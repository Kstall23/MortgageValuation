import numpy as np
import pandas as pd
import tensorflow
import sklearn.preprocessing as pre


print("Hey look, maybe someday this will provide a prediction point.")

# read in training data file as a pandas dataframe, assigning appropriate headers
# --- Preprocess Breast Cancer Data -----------------
''' attribute values discrete (1-10, normalized values, ordinal)'''
# read in data file as a dataframe, assigning appropriate headers
cancer = pd.read_csv('backend\\machineLearning\\breast-cancer-wisconsin.data', sep=',', names=["id", "clumpThickness", "unifCellSize", 
         "unifCellShape", "adhesion", "singelEpithSize", "bareNuclei", "blandChrom", "normalNucleoli", "mitoses", "class"])

# remove missing information, only 16 so remove whole row
for i in range(len(cancer["bareNuclei"])):
    if (cancer["bareNuclei"][i] == '?'):
        cancer.drop(i,inplace=True,axis=0)


# separate into X and y numpy arrays, drop column 0 (id)
X = cancer.iloc[:,1:10].values
y = cancer.iloc[:,10:11].values

# Normalize the data
# print(X)
# sc = pre.StandardScaler()
# X - sc.fit_transform(X)
# print(X)

# ohe = pre.OneHotEncoder()
# y = ohe.fit_transform(y).toarray()
X = np.asarray(X).astype(np.float32)
y = np.asarray(y).astype(np.float32)
y = y/2 -1

print(X)
print(y)

# Build the model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=9, activation="sigmoid"))     # first hidden layer, 10 nodes
model.add(Dense(10, activation="sigmoid"))                  # second hidden layer, 10 nodes
model.add(Dense(1, activation="sigmoid"))                   # output layer, 1 node

# compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(X, y, epochs=150, batch_size=30)

# Evaluate the model
_, accuracy = model.evaluate(X, y)
print("Model accuracy: %.2f" % (accuracy*100) + "%")


new_point = np.asarray([[7, 1, 5, 1, 4, 8, 2, 3, 5]]).astype(np.float32)
prediction = model.predict(new_point)
print([round(x[0]) for x in prediction])



