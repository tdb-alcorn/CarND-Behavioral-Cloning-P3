from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda, Cropping2D

import numpy as np
import csv
import cv2

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(110,300,3)))

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
    with open('/opt/data/driving_log.csv', 'r') as f:
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

def create_generators(data, validation_split=0.2):
    np.random.shuffle(data)
    valid_split_idx = int((1 - validation_split) * len(data))
    training_data = data[:valid_split_idx]
    validation_data = data[valid_split_idx:]
    def generator_wrapper(x_data):
        def wrapped():
            for row in data:
                img_file = row['center']
                steering = row['steering']
                img = cv2.imread('/opt/data/' + img_file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                yield img, steering
        return wrapped
    training_generator = generator_wrapper(training_data)
    validation_generator = generator_wrapper(validation_data)
    return training_generator, validation_generator, len(training_data), len(validation_data)

data = load_data()
training_generator, validation_generator, num_training_samples, num_validation_samples = create_generators(data)

model.compile('adam', 'mse', ['accuracy'])
history = model.fit_generator(training_generator, steps_per_epoch =
    num_training_samples/10, validation_data = 
    validation_generator,
    validation_steps = num_validation_samples/10,
    nb_epoch=1, verbose=1)

metrics = model.evaluate(X, y)

print(history)
print(metrics)

model.save('model.h5')