from pathlib import Path

general = {
    "dataset_path": Path("../dataset/fer2013/fer2013/fer2013.csv"),
    "pickle_path": Path("../pickels"),
    "pickle_data_path": Path("../pickels/emotion_recognition"),
    "pickle_history_path": Path("../pickels/emotion_recognition/history")
}

# example
cnn_0 = {
    "id": "cnn_0",
    "epochs": 2,
    "optimizer": "Adam",
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy"]
}

