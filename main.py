import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical
import config
import time
from pathlib import Path
import pickle

def create_dir(path):
    path = str(path)
    if os.path.exists(path):
        print("Directory %s exists" % path)
        return True
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
        return False
    else:
        print("Successfully created the directory %s " % path)
        return True

def picke_load(pickle_path):
    if pickle_path.exists():
        pickle_in = open(str(pickle_path), "rb")
        return pickle.load(pickle_in)
    else:
        return None

def pickle_save(object, pickle_path):
    pickle_out = open(str(pickle_path), "wb")
    pickle.dump(object, pickle_out)
    pickle_out.close()

def extract_data():
    df = pd.read_csv(config.general["dataset_path"])

    training     = df.loc[df["Usage"] == "Training"]
    private_test = df.loc[df["Usage"] == "PrivateTest"]
    public_test  = df.loc[df["Usage"] == "PublicTest"]

    def preprocess_data(data):
        labels = to_categorical(data["emotion"])
        pixels = data["pixels"].str.split(" ").tolist()
        pixels = np.uint8(pixels)
        pixels = pixels.reshape((len(data["emotion"]), 48, 48, 1))
        pixels = pixels.astype("float32") / 255
        return labels, pixels

    training_labels, training_pixels         = preprocess_data(training)
    private_test_labels, private_test_pixels = preprocess_data(private_test)
    public_test_labels, public_test_pixels   = preprocess_data(public_test)

    training_data     = (training, training_labels, training_pixels)
    private_test_data = (private_test, private_test_labels, private_test_pixels)
    public_test_data  = (public_test, public_test_labels, public_test_pixels)
    return training_data, private_test_data, public_test_data

def main():
    start = time.time()
    # get picke paths
    pickle_training    = Path(config.general["pickle_data_path"]) / "training_data.pickle"
    picle_private_test = Path(config.general["pickle_data_path"]) / "private_test_data.pickle"
    picle_public_test  = Path(config.general["pickle_data_path"]) / "public_test_data.pickle"

    # try to load the pickels
    training_data     = picke_load(pickle_training)
    private_test_data = picke_load(picle_private_test)
    public_test_data  = picke_load(picle_public_test)

    # if there is no pickels extract datasets from csv, transform them and save them in pickels
    if training_data == None or private_test_data == None or public_test_data == None:

        create_dir(config.general["pickle_path"])
        create_dir(config.general["pickle_data_path"])

        training_data, private_test_data, public_test_data = extract_data()

        pickle_save(training_data, pickle_training)
        pickle_save(private_test_data, picle_private_test)
        pickle_save(public_test_data, picle_public_test)

    training, training_labels, training_pixels             = training_data
    private_test, private_test_labels, private_test_pixels = private_test_data
    public_test, public_test_labels, public_test_pixels    = public_test_data

    print("Extraction and preprocessing time: ", str(time.time() - start))

    for cnn in config.cnn_list:
        start = time.time()
        # init cnn
        model = models.Sequential()

        # convolution
        model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
        model.add(layers.MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

        model.add(layers.Flatten())

        # add layers
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.2))


        model.add(layers.Dense(7, activation='softmax'))

        # launch cnn
        model.compile(
            optimizer = cnn["optimizer"],
            loss = cnn["loss"],
            metrics = cnn["metrics"]
        )
        print(model.summary())
        hist = model.fit(training_pixels, training_labels, batch_size = 256, epochs = cnn["epochs"],
                        validation_data = (private_test_pixels, private_test_labels))

        print("CNN training time: ", str(time.time() - start))

        # get results
        training_accuracy = hist.history["accuracy"]
        validation_accuracy = hist.history["val_accuracy"]

        training_loss = hist.history["loss"]
        validation_loss = hist.history["val_loss"]

        epochs = range(1, cnn["epochs"] + 1)

        plt.plot(epochs, training_accuracy, "b", label = "Training Success")
        plt.plot(epochs, validation_accuracy, "r", label = "Validation Success")
        plt.title("Training and Validation results")
        plt.legend()

        plt.figure()

        plt.plot(epochs, training_loss, "b", label = "Training Loss")
        plt.plot(epochs, validation_loss, "r", label = "Validation Loss")
        plt.title("Training and Validation losses")
        plt.legend()

        plt.show()

main()
