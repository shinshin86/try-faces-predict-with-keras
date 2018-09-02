from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from PIL import Image
import keras
import os
import glob
import numpy as np
from sklearn import model_selection

# Parameters is for development
TARGET_PATH = 'detect_images'
NPY_FILE_NAME = 'test.npy'
H5_FILE_NAME = 'test.h5'
GEN_DATA_COUNT = 100
IMAGE_SIZE = 50
BATCH_SIZE = 32
EPOCHS_COUNT = 10

def initialize():
    classlist = glob.glob(os.path.join(TARGET_PATH, '*'))
    classes = []
    for i in classlist:
        classes.append(os.path.basename(i))
    return classes


def main(classes):
    num_classes = len(classes)
    X_train, X_test, y_train, y_test = np.load(NPY_FILE_NAME)
    X_train = X_train.astype('float') / 256
    X_test = X_test.astype('float') / 256
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)

def gen_data(classes):
    X = []
    Y = []
    for index, label in enumerate(classes):
        photos_dir = os.path.join(TARGET_PATH, label)
        files = glob.glob(photos_dir + '/*.jpg')
        for i, file in enumerate(files):
            if i >= GEN_DATA_COUNT:
                break
            image = Image.open(file)
            image = image.convert('RGB')
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            data = np.asarray(image)
            X.append(data)
            Y.append(index)

    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
    xy = (X_train, X_test, y_train, y_test)
    np.save(NPY_FILE_NAME, xy)

def model_train(X, y):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0, 25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0, 25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS_COUNT)

    model.save(H5_FILE_NAME)

    return model


def model_eval(model, X, y):
    scores = model.evaluate(X, y, verbose=1)
    print('Test Loss: ', scores[0])
    print('Test Accuracy: ', scores[1])


if __name__ == '__main__':
    classes = initialize()
    gen_data(classes)
    main(classes)
