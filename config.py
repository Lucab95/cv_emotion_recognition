from pathlib import Path

general = {
    "dataset_path": Path("../dataset/fer2013/fer2013/fer2013.csv"),
    "pickle_path": Path("../pickels"),
    "pickle_data_path": Path("../pickels/emotion_recognition")
}

# example
cnn = {
    "epochs": 5,
    "optimizer": "Adam",
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy"]
}

cnn_0 = {

}

cnn_list = [cnn]