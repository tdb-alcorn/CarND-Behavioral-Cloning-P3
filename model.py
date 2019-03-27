from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda, Cropping2D

import numpy as np
import csv
import cv2

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

model.summary()

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
    with open('./data/driving_log.csv', 'r') as f:
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
    np.random.shuffle(data)
    test_idx = int((1 - test_split) * len(data))
    test_data = data[test_idx:]
    data = data[:test_idx]
    valid_split_idx = int((1 - validation_split)/(1 - test_split) * len(data))
    training_data = data[:valid_split_idx]
    validation_data = data[valid_split_idx:]
    def gen(x_data):
        i = 0
        X = np.zeros((batch_size, 160, 320, 3))
        y = np.zeros((batch_size,))
        for row in x_data:
            if i == batch_size:
                yield X, y
                i = 0
                X = np.zeros((batch_size, 160, 320, 3))
                y = np.zeros((batch_size,))
            img_file = row['center']
            steering = row['steering']
            img = cv2.imread('./data/' + img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            # img = cv2.resize(img, (66, 200))
            X[i] = img
            y[i] = steering
            i += 1
    training_generator = gen(training_data)
    validation_generator = gen(validation_data)
    test_generator = gen(test_data)
    return training_generator, validation_generator, test_generator, len(training_data), len(validation_data), len(test_data)

data = load_data()

training_generator, validation_generator, test_generator, num_training_samples, num_validation_samples, num_test_samples = create_generators(data, 10)
print(validation_generator)

model.compile('adam', 'mse', ['accuracy'])
history = model.fit_generator(
    training_generator,
    # steps_per_epoch=num_training_samples//10,
    steps_per_epoch=10,
    validation_data=validation_generator,
    # validation_steps=num_validation_samples//10,
    validation_steps=10,
    epochs=1,
    verbose=1,
    )

metrics = model.evaluate_generator(test_generator, steps=10)

print(history)
print(metrics)

model.save('model.h5')