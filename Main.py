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
#from sklearn.metrics import confusion_matrix
#import itertools

from tensorflow.python.keras.utils.np_utils import to_categorical 
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

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

# Encode the labels to vector (ex: 5 -> [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
y_train = to_categorical(y_train, num_classes = 10)

# Set the random seed and split the training and validation set
random_seed = 2
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = random_seed)

# Set the CNN model
# The architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'Same', activation = 'relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model = Sequential()
model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), strides = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

# Compile the model
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics  = ["accuracy"])

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

# Set the epochs and batch_size
epochs = 3
batch_size = 86

# Use data augmentation to prevent overfitting
datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             rotation_range=15,
                             zoom_range=0.1,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=False,
                             vertical_flip=False)
datagen.fit(X_train)

# Fit the model
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size),
                              epochs = epochs,
                              validation_data = (X_val, y_val),
                              verbose = 2,
                              steps_per_epoch = X_train.shape[0],
                              callbacks = [learning_rate_reduction])

# Plot the loss and accuracy curves for both training and validation sets
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color = 'b', label = "Training loss")
ax[0].plot(history.history['val_loss'], color = 'r', label = "Validation loss", axes = ax[0])
legend = ax[0].legend(loc = 'best', shadow = True)

ax[1].plot(history.history['acc'], color = 'b', label = "Training accuracy")
ax[1].plot(history.history['val_acc'], color = 'r', label = "Validation accuracy")
legend = ax[1].legend(loc = 'best', shadow = True)

# Predict results
results = model.predict(test)

# Select the index with the maximum probability
results = np. argmax(results, axis = 1)
results = pd.Series(results, name = "Label")
submission = pd.concat([pd.Series(range(1, 28001), name = "ImageId"), results], axis = 1)
submission.to_csv("mnist_submission.csv", index = False)