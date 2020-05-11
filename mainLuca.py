# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow
# from tensorflow import keras
from tensorflow.keras import layers
# from keras.utils import np_utils
from tensorflow.python.client import device_lib

# print(device_lib.list_local_devices())
print(tensorflow.__version__)
tensorflow.test.is_built_with_cuda()

df = pd.read_csv('../fer2013/fer2013/fer2013.csv')
# df.info()
# print(df.head())
# print("righe,colonne = ", df.shape)
# print("Column name= ", df.columns)



training = df.loc[df["Usage"] == "Training"]
# print(training)

emotions = training["emotion"]
print(emotions)
train_emotions = np_utils.to_categorical(emotions)
print(train_emotions)
pixel = training["pixels"].str.split(" ").tolist()
pixel = np.uint8(pixel)
print("ok")
print("pixel",pixel)
pixel = pixel.reshape(pixel.shape[0],48,48)
pixel = pixel.astype("float32") / 255



private = df.loc[df["Usage"] == "PrivateTest"]
private_labels = private["emotion"]
private_labels = np_utils.to_categorical(private_labels)

private_pixels = private["pixels"].str.split(" ").tolist()
private_pixels = np.uint8(private_pixels)
private_pixels = private_pixels.reshape((private_pixels.shape[0], 48, 48, 1))
private_pixels = private_pixels.astype("float32") / 255



#define a sequential model
seq_model = keras.Sequential()
seq_model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
seq_model.summary()

seq_model.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
seq_model.add(layers.MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

seq_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
seq_model.add(layers.AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

seq_model.add(layers.Flatten())

#final
seq_model.add(layers.Dense(1024, activation='relu'))
seq_model.add(layers.Dropout(0.2))
seq_model.add(layers.Dense(1024, activation='relu'))
seq_model.add(layers.Dropout(0.2))

seq_model.add(layers.Dense(7, activation='softmax'))
seq_model.summary()

seq_model.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
hist = seq_model.fit(pixel, emotions, batch_size = 256, epochs = 30,
                validation_data = (private_pixels, private_labels))

# img = df["pixels"][200]
# val = img.split(" ")
# x_pixels = np.array(val, 'float32')
# x_pixels /= 255
# print(x_pixels)
# x_reshaped = x_pixels.reshape(48,48)