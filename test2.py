import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import input

(X_train, y_train, frame, cate) = input.load_data("wav")
(X_test, y_test, t_frame, cate) = input.load_data("test")

# data pre-processing
X_train = X_train.reshape(-1, 1, frame, 20)
X_test = X_test.reshape(-1, 1, t_frame, 20)
y_train = np_utils.to_categorical(y_train, num_classes=cate)
y_test = np_utils.to_categorical(y_test, num_classes=cate)


model = Sequential()

# Convlayer
model.add(Convolution2D(
    batch_input_shape=(None, 1, frame, 20),
    filters=20,
    kernel_size=(2, 1),
    strides=1,
    padding='same',
    data_format='channels_first',
))
model.add(Activation('relu'))

# Pooling layer
model.add(MaxPooling2D(
    pool_size=(2, 1),
    strides=2,
    padding='same',
    data_format='channels_first',
))

# Fully connected layer
model.add(Flatten())

model.add(Dense(cate))
model.add(Activation('softmax'))

adam = Adam(lr=1e-4)

model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')

model.fit(X_train, y_train, epochs=10, batch_size=12,)

print('\nTesting ------------')

loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

print("\nTraning round 2--------")
print(X_test.shape)
print(y_test.shape)
model.train_on_batch(X_test, y_test)

print('\nTesting ------------')

loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

with open("predict_data", "w") as f:
    f.write(str(model.predict(X_test)))
    f.close()



