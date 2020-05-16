from keras import models
from keras import layers
import config
import time
import janitor as jn
import tensorflow as tf

from tensorflow.keras import datasets, layers, models



def cnn_0(training_data, private_test_data, public_test_data):
    start = time.time()

    cnn = {
        "id": "cnn_0",
        "epochs": 30,
        "optimizer": "Adam",
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"]
    }

    training, training_labels, training_pixels = training_data
    private_test, private_test_labels, private_test_pixels = private_test_data
    public_test, public_test_labels, public_test_pixels = public_test_data

    # init cnn
    model = models.Sequential()

    # convolution
    model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
    model.add(layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))


    # add layers
    model.add(layers.Flatten())

    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(7, activation='softmax'))

    model.summary()

    # launch cnn
    model.compile(
        optimizer=cnn["optimizer"],
        loss=cnn["loss"],
        metrics=cnn["metrics"]
    )

    hist = model.fit(training_pixels,
                     training_labels,
                     batch_size=256,
                     epochs=cnn["epochs"],
                     validation_data=(private_test_pixels, private_test_labels),
                     verbose=2
                     )

    print("CNN training time: ", str(time.time() - start))

    test_loss, test_accuracy = model.evaluate(public_test_pixels, public_test_labels, verbose=0)
    print("public test acc  -> ", test_accuracy)
    print("public test loss -> ", test_loss)

    jn.create_dir(config.general["pickle_history_path"])
    saving_history_path = config.general["pickle_history_path"] / str("history_" + cnn["id"] + ".pickle")
    history = {
        "id": cnn["id"],
        "epochs": cnn["epochs"],
        "history": hist.history,
        "test": [test_accuracy, test_loss]
    }
    jn.pickle_save(history, saving_history_path)


def cnn_1(training_data, private_test_data, public_test_data):
    start = time.time()

    cnn = {
        "id": "cnn_1",
        "epochs": 30,
        "optimizer": "Adam",
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"]
    }

    training, training_labels, training_pixels = training_data
    private_test, private_test_labels, private_test_pixels = private_test_data
    public_test, public_test_labels, public_test_pixels = public_test_data

    # init cnn
    model = models.Sequential()

    # convolution
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))



    # add layers
    model.add(layers.Flatten())

    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(7, activation='softmax'))


    model.summary()

    model.compile(optimizer='adam',
                  loss=cnn["loss"],
                  metrics=['accuracy']
                  )

    hist = model.fit(training_pixels,
                     training_labels,
                     epochs=cnn["epochs"],
                     validation_data=(private_test_pixels, private_test_labels),
                     verbose=2
                     )

    print("CNN training time: ", str(time.time() - start))


    test_loss, test_accuracy = model.evaluate(public_test_pixels, public_test_labels, verbose=0)
    print("public test acc  -> ", test_accuracy)
    print("public test loss -> ", test_loss)

    jn.create_dir(config.general["pickle_history_path"])
    saving_history_path = config.general["pickle_history_path"] / str("history_" + cnn["id"] + ".pickle")
    history = {
        "id": cnn["id"],
        "epochs": cnn["epochs"],
        "history": hist.history,
        "test": [test_accuracy, test_loss]
    }
    jn.pickle_save(history, saving_history_path)


def cnn_2(training_data, private_test_data, public_test_data):
    pass



