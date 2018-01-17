import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import input

(x1, y1, frame, cate) = input.load_data("group0")
print(frame, cate)
(x2, y2, frame, cate) = input.load_data("group1")
print(frame, cate)
(x3, y3, frame, cate) = input.load_data("group2")
print(frame, cate)
(x4, y4, frame, cate) = input.load_data("group3")
print(frame, cate)
(x5, y5, frame, cate) = input.load_data("group4")
print(frame, cate)
x = []
y = []
x.append(x1.reshape(-1, 1, frame, 20))
x.append(x2.reshape(-1, 1, frame, 20))
x.append(x3.reshape(-1, 1, frame, 20))
x.append(x4.reshape(-1, 1, frame, 20))
x.append(x5.reshape(-1, 1, frame, 20))
y.append(np_utils.to_categorical(y1, num_classes=cate))
y.append(np_utils.to_categorical(y2, num_classes=cate))
y.append(np_utils.to_categorical(y3, num_classes=cate))
y.append(np_utils.to_categorical(y4, num_classes=cate))
y.append(np_utils.to_categorical(y5, num_classes=cate))

f = open("5fold_test.txt", "w")

for i in range(0, 5):
    X_test = x[i]
    y_test = y[i]
    print(X_test.shape, y_test.shape)
    X_train = np.vstack((x[i - 1], x[i - 2], x[i - 3], x[i - 4]))
    y_train = np.vstack((y[i - 1], y[i - 2], y[i - 3], y[i - 4]))
    print(X_train.shape, y_train.shape)
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

    model.fit(X_train, y_train, epochs=30)

    print('\nTesting ------------')

    loss, accuracy = model.evaluate(X_test, y_test)

    print('\ntest loss: ', loss)
    print('\ntest accuracy: ', accuracy)
    f.write("\nRound" + str(i))
    f.write('\ntest loss: ' + str(loss))
    f.write('\ntest accuracy: '+ str(accuracy))

f.close()

#print("\nTraning round 2--------")

#model.fit(X_test, y_test, epochs=3)

#print('\nTesting ------------')

#loss, accuracy = model.evaluate(X_test, y_test)

#print('\ntest loss: ', loss)
#print('\ntest accuracy: ', accuracy)

#with open("predict_data", "w") as f:
#    f.write(str(model.predict(X_test)))
#    f.close()



