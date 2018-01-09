import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import input

(X_train, y_train, frame, cate) = input.load_data("wav")
(X_test, y_test, frame, cate) = input.load_data("wav")

# data pre-processing
X_train = X_train.reshape(-1, 1, frame, 14)
X_test = X_test.reshape(-1, 1, frame, 14)
y_train = np_utils.to_categorical(y_train, num_classes=cate)
y_test = np_utils.to_categorical(y_test, num_classes=cate)

# Another way to build your CNN
model = Sequential()

# Conv layer 1 output shape (32, 253, 14)
model.add(Convolution2D(
    batch_input_shape=(None, 1, frame, 14),
    filters=32,
    kernel_size=(2, 1),
    strides=1,
    padding='same',     # Padding method
    data_format='channels_first',
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (32, 126, 7)
model.add(MaxPooling2D(
    pool_size=(2, 1),
    strides=2,
    padding='same',    # Padding method
    data_format='channels_first',
))

<<<<<<< HEAD
# Conv layer 2 output shape
model.add(Convolution2D(64, (2, 1), strides=1, padding='same', data_format='channels_first'))
model.add(Activation('softmax'))

# Pooling layer 2 (max pooling) output shape
model.add(MaxPooling2D((2, 1), 2, 'same', data_format='channels_first'))

# Conv layer 3 output shape
model.add(Convolution2D(128, (2, 1), strides=1, padding='same', data_format='channels_first'))
model.add(Activation('softmax'))

# Pooling layer 3 (max pooling) output shape (64, 63, 3)
model.add(MaxPooling2D((2, 1), 2, 'same', data_format='channels_first'))

# Conv layer 3 output shape
model.add(Convolution2D(256, (2, 1), strides=1, padding='same', data_format='channels_first'))
model.add(Activation('softmax'))

# Pooling layer 3 (max pooling) output shape (64, 63, 3)
model.add(MaxPooling2D((2, 1), 2, 'same', data_format='channels_first'))
=======
# Conv layer 2 output shape (64, 126, 7)
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 63, 3)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))
>>>>>>> parent of 4258c37... 180109

# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('softmax'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(25))
model.add(Activation('softmax'))

model.add(Dense(cate))
model.add(Activation('softmax'))


# Another way to define your optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
<<<<<<< HEAD
model.fit(X_train, y_train, epochs=5, batch_size=12,)
=======
model.fit(X_train, y_train, epochs=1, batch_size=64,)
>>>>>>> parent of 4258c37... 180109

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)