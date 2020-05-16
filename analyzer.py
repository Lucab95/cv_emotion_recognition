from pathlib import Path
from os import listdir
from os.path import isfile, join
import re
from os import path
import matplotlib.pyplot as plt
import time
import pickle
import config

def picke_load(pickle_path):
    if pickle_path.exists():
        pickle_in = open(str(pickle_path), "rb")
        return pickle.load(pickle_in)
    else:
        return None

def cnn_analysis():
    history_path = config.general["pickle_history_path"]
    file_list = [f for f in listdir(history_path) if isfile(join(history_path, f))]
    history_list = []
    for file_name in file_list:
        correct_regex = re.match("\Ahistory_(.*)\.pickle\Z", file_name)
        if not correct_regex:
            print(file_name, " not considered")
        else:
            hist = picke_load(config.general["pickle_history_path"] / file_name)
            history_list.append(hist)

    for hist in history_list:
        id = hist["id"]

        epochs = range(1, hist["epochs"] + 1)

        plt.plot(epochs, hist["history"]["accuracy"], "b", label="Training Success")
        plt.plot(epochs, hist["history"]["val_accuracy"], "r", label="Validation Success")
        plt.title("Training and Validation results")
        plt.legend()

        plt.figure()

        plt.plot(epochs, hist["history"]["loss"], "b", label="Training Loss")
        plt.plot(epochs, hist["history"]["val_loss"], "r", label="Validation Loss")
        plt.title("Training and Validation losses")
        plt.legend()

        plt.show()


cnn_analysis()
