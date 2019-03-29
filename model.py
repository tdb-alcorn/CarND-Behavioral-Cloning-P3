import os
import csv

import numpy as np
import cv2

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda, Cropping2D


data_dir = './data/'
# data_dir = './run0/'
# data_dir = './run1/'

num_epochs = 5
batch_size = 10


def create_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(90, 320, 3)))

    model.add(Conv2D(24, 5, strides=(2,2), activation='relu'))
    model.add(Conv2D(36, 5, strides=(2,2), activation='relu'))
    model.add(Conv2D(48, 5, strides=(2,2), activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))

    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation=None))

    return model


def flip_image(image, measurement):
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    return image_flipped, measurement_flipped


def read_row(row, header):
    row_dict = dict()
    for i in range(len(header)):
        row_dict[header[i]] = row[i]
    return row_dict


def load_data():
    data = list()
    with open(data_dir + 'driving_log.csv', 'r') as f:
        reader = csv.reader(f)
        first = True
        header = None
        for row in reader:
            if first:
                header = row
                first = False
            else:
                data.append(read_row(row, header))
    return data


def create_generators(data, batch_size, validation_split=0.2, test_split=0.1):
    np.random.seed(42)
    np.random.shuffle(data)
    np.random.seed()
    test_idx = int((1 - test_split) * len(data))
    test_data = data[test_idx:]
    data = data[:test_idx]
    valid_split_idx = int((1 - validation_split)/(1 - test_split) * len(data))
    training_data = data[:valid_split_idx]
    validation_data = data[valid_split_idx:]

    def gen(x_data):
        epoch_num = 0
        batch_num = 0
        num_samples = len(x_data)
        X = np.zeros((batch_size, 160, 320, 3))
        y = np.zeros((batch_size,))
        while True:
            if batch_num == batch_size:
                yield X, y
                batch_num = 0
                X = np.zeros((batch_size, 160, 320, 3))
                y = np.zeros((batch_size,))
            row = x_data[epoch_num]
            img_file = '/'.join(row['center'].split('/')[-2:])
            steering = float(row['steering'])
            img = cv2.imread(data_dir + img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            # img = cv2.resize(img, (66, 200))
            if np.random.random() < 0.5:
                img, steering = flip_image(img, steering)
            X[batch_num] = img
            y[batch_num] = steering
            batch_num += 1
            epoch_num += 1
            epoch_num = epoch_num % num_samples

    training_generator = gen(training_data)
    validation_generator = gen(validation_data)
    test_generator = gen(test_data)

    return training_generator, validation_generator, test_generator, \
        len(training_data)//batch_size, len(validation_data)//batch_size, \
        len(test_data)//batch_size


if __name__ == '__main__':
    data = load_data()

    training_generator, \
    validation_generator, \
    test_generator, \
    num_training_steps, \
    num_validation_steps, \
    num_test_steps = create_generators(data, batch_size)

    if os.path.isfile('./model.h5'):
        print('loading model from model.h5')
        model = load_model('./model.h5')
    else:
        print('creating model')
        model = create_model()
        model.compile('adam', 'mse', ['accuracy'])

    model.summary()

    history = model.fit_generator(
        training_generator,
        steps_per_epoch=num_training_steps,
#         steps_per_epoch=10,
        validation_data=validation_generator,
        validation_steps=num_validation_steps,
#         validation_steps=10,
        epochs=num_epochs,
        verbose=1,
        )

    metrics = model.evaluate_generator(test_generator,
        steps=num_test_steps,
#         steps=10,
    )

    print(history)
    print(metrics)

    model.save('model.h5')