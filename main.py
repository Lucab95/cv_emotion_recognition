import pandas as pd
import numpy as np
from keras.utils import to_categorical
import config
import cnn
import time
import janitor as jn


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
    pickle_training    = config.general["pickle_data_path"] / "training_data.pickle"
    picle_private_test = config.general["pickle_data_path"] / "private_test_data.pickle"
    picle_public_test  = config.general["pickle_data_path"] / "public_test_data.pickle"

    # try to load the pickels
    training_data     = jn.picke_load(pickle_training)
    private_test_data = jn.picke_load(picle_private_test)
    public_test_data  = jn.picke_load(picle_public_test)

    # if there is no pickels extract datasets from csv, transform them and save them in pickels
    if training_data == None or private_test_data == None or public_test_data == None:

        jn.create_dir(config.general["pickle_path"])
        jn.create_dir(config.general["pickle_data_path"])

        training_data, private_test_data, public_test_data = extract_data()

        jn.pickle_save(training_data, pickle_training)
        jn.pickle_save(private_test_data, picle_private_test)
        jn.pickle_save(public_test_data, picle_public_test)

    print("Extraction and preprocessing time: ", str(time.time() - start))

    # launch a list of cnn
    #cnn.cnn_0(training_data, private_test_data, public_test_data)
    cnn.cnn_1(training_data, private_test_data, public_test_data)

main()
