import os
import csv
import matplotlib.image as mpimg
import numpy as np
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

#Loading training data from csv file and image files
X_data = []
y_data = []
with open('driving_log.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in csv_reader:
        filename = row[0]
        wheel_angle = row[3]
        img = mpimg.imread(filename).astype('uint8')
        #Cropping image to extract only the road ahead 
        crop_image = img[80:140, 0:320]       
        #Resizing image to reduce memory consuption 
        res_img = imresize(crop_image, (32,32,3))
        #Adding image and wheel angle data to training set
        X_data.append(res_img)
        y_data.append(float(wheel_angle))

print("Training data loaded.")

#Normalizing features
X_norm =  (np.array(X_data) - 127.) / 128.
print("Training data normalized.")

#Shuffling train data, creating test set
X_train, X_test, y_train, y_test = train_test_split(
   X_norm,
   y_data,
   test_size=0.2,
   random_state=42
)

print("Shape of training set: " + str(X_train.shape))

# Fix error with TF and Keras
tf.python.control_flow_ops = tf

# Creating model architecture
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('tanh'));

#Compiling model with Adam optimizer
model.compile(loss='mean_squared_error',
              optimizer=Adam(lr = 0.001),
              metrics=['accuracy'])

# Training model
history = model.fit(X_train, y_train, batch_size = 128, nb_epoch=15, validation_split=0.2)

# Saving model to JSON file
json = model.to_json()
with open('model.json', 'w') as f:
        f.write(json)
model.save_weights('model.h5')

# Evaluating model on test data
print('Evaluating model on test data')
metrics = model.evaluate(X_test, y_test)
print('\nTest data evaluation results:')
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))