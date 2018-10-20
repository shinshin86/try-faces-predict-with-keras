from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from PIL import Image
import keras
import os
import sys
import glob
import numpy as np


# using constant
TARGET_PATH = 'detect_images'
H5_FILE_NAME = 'test.h5'
image_size = 50


def get_classes():
    classlist = glob.glob(os.path.join(TARGET_PATH, '*'))
    classes = []
    for i in classlist:
        classes.append(os.path.basename(i))
    return classes


def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(50, 50, 3)))
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

    model = load_model(H5_FILE_NAME)

    return model


def main():
    classes = get_classes()
    image = Image.open(sys.argv[1])
    image = image.convert('RGB')
    image = image.resize((image_size, image_size))
    data = np.asarray(image)

    X = []
    X.append(data)
    X = np.array(X)
    model = build_model()

    result = model.predict([X])[0]
    predicted = result.argmax()
    percentage = str(result[predicted] * 100)
    print("Result (Accuracy) ============>  {0} ({1} %)".format(classes[predicted], percentage))


if __name__ == '__main__':
    main()
