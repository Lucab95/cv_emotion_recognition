from os import listdir
from os.path import isfile, join
import re
import matplotlib.pyplot as plt
import config
import janitor as jn


def cnn_analysis():
    history_path = config.general["pickle_history_path"]
    file_list = [f for f in listdir(history_path) if isfile(join(history_path, f))]
    history_list = []
    for file_name in file_list:
        correct_regex = re.match("\Ahistory_(.*)\.pickle\Z", file_name)
        if not correct_regex:
            print(file_name, " not considered")
        else:
            hist = jn.picke_load(config.general["pickle_history_path"] / file_name)
            history_list.append(hist)

    result_list = []
    for hist in history_list:
        id = hist["id"]
        epochs = range(1, hist["epochs"] + 1)

        plt.plot(epochs, hist["history"]["loss"], "b", label="Training Loss")
        plt.plot(epochs, hist["history"]["val_loss"], "r", label="Validation Loss")
        plt.title(str("Training and Validation losses of " + id))
        plt.legend()

        plt.figure()

        plt.plot(epochs, hist["history"]["accuracy"], "b", label="Training Success")
        plt.plot(epochs, hist["history"]["val_accuracy"], "r", label="Validation Success")
        plt.title(str("Training and Validation results of " + id))
        plt.legend()

        plt.show()

        print("Public Test ", id, " accuracy: ", hist["test"][0])
        print("Public Test ", id, " loss: ", hist["test"][1])

        result_list.append([id, hist["history"]["val_accuracy"], hist["history"]["val_loss"]])

    plt.title(str("Accuracy results"))
    color_list = ["blue", "red", "green", "orange"]
    for i, res in enumerate(result_list):
        id = res[0]
        acc = res[1]

        plt.plot(epochs, acc, color_list[i], label=id)
        plt.legend()

    plt.show()

    plt.title(str("Losses loss"))
    color_list = ["blue", "red", "green", "orange"]
    for i, res in enumerate(result_list):
        id = res[0]
        los = res[2]

        plt.plot(epochs, los, color_list[i], label=id)
        plt.legend()

    plt.show()


cnn_analysis()
