import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from keras import models
from keras import layers
from keras.utils import np_utils
from keras.utils import to_categorical
import pickle
from tensorflow.python.client import device_lib

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

tf.debugging.set_log_device_placement(True)

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)

# print(device_lib.list_local_devices())
print(tensorflow.__version__)
tensorflow.test.is_built_with_cuda()

df = pd.read_csv('../fer2013/fer2013/fer2013.csv')
# df.info()
# print(df.head())
# print("righe,colonne = ", df.shape)
# print("Column name= ", df.columns)


try:
    train_emotion = pickle.load(open("train_emotion.p","rb"))
    train_pixel = pickle.load(open("train_pixel.p","rb"))
    private_emotion = pickle.load(open("private_emotion.p","rb"))
    private_pixel = pickle.load(open("private_pixel.p","rb"))
    print(train_emotion)
except:
    print("no file")




    training = df.loc[df["Usage"] == "Training"]
    public_test = df.loc[df["Usage"] == "PublicTest"]
    private_test = df.loc[df["Usage"] == "PrivateTest"]


    train_emotion = training["emotion"]
    train_emotion = to_categorical(train_emotion)

    train_pixel = training["pixels"].str.split(" ").tolist()
    train_pixel = np.uint8(train_pixel)
    train_pixel = train_pixel.reshape((28709, 48, 48, 1))
    train_pixel = train_pixel.astype("float32") / 255


    private_emotion = private_test["emotion"]
    private_emotion = to_categorical(private_emotion)

    private_pixel = private_test["pixels"].str.split(" ").tolist()
    private_pixel = np.uint8(private_pixel)
    private_pixel = private_pixel.reshape((3589, 48, 48, 1))
    private_pixel = private_pixel.astype("float32") / 255




    public_emotion = public_test["emotion"]
    public_emotion = to_categorical(public_emotion)

    public_pixels = public_test["pixels"].str.split(" ").tolist()
    public_pixels = np.uint8(public_pixels)
    public_pixels = public_pixels.reshape((3589, 48, 48, 1))
    public_pixels = public_pixels.astype("float32") / 255

    pickle.dump(train_emotion, open("train_emotion.p", "wb"))
    pickle.dump(train_pixel, open("train_pixel.p", "wb"))
    pickle.dump(private_emotion, open("private_emotion.p", "wb"))
    pickle.dump(private_pixel, open("private_pixel.p", "wb"))
    # pickle.dump(public_pixels, open("public_pixel.p", "wb"))
    # pickle.dump(public_emotion, open("public_emotion.p", "wb"))

# private = df.loc[df["Usage"] == "PrivateTest"]
# private_labels = private["emotion"]
# private_labels = np_utils.to_categorical(private_labels)
#
# private_pixels = private["pixels"].str.split(" ").tolist()
# private_pixels = np.uint8(private_pixels)
# private_pixels = private_pixels.reshape((private_pixels.shape[0], 48, 48, 1))
# private_pixels = private_pixels.astype("float32") / 255



model = models.Sequential()

# model = models.Sequential()

# Conv (evrişim katmanı)
model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
#Ortaklama katmanı
model.add(layers.MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
#
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
#
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
#
model.add(layers.Flatten())

# Tam bağlantı katmanı
# model.add(layers.Dense(1024, activation='relu'))
# model.add(layers.Dropout(0.2))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(7, activation='softmax'))

# model.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
# model.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


hist = model.fit(train_pixel, train_emotion, batch_size = 256, epochs = 5,
                validation_data = (private_pixel, private_emotion))

# img = df["pixels"][200]
# val = img.split(" ")
# x_pixels = np.array(val, 'float32')
# x_pixels /= 255
# print(x_pixels)
# x_reshaped = x_pixels.reshape(48,48)