'''
Created on May 4, 2019

@author: Yawen
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

sns.set(style = 'white', context = 'notebook', palette = 'deep')

# First import the data from csv files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y_train = train["label"]
X_train = train.drop(["label"], axis = 1)
del train
g = sns.countplot(y_train)
plt.show()
print(y_train.value_counts())

# Check for null and missing values
print(X_train.isnull().any().describe())
print(test.isnull().any().describe())

# Normalize and reshape the data
X_train = X_train / 255.0
test = test / 255.0
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1,28,28,1)


