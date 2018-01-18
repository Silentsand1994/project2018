import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from pro_class import EmotionData

dataload = EmotionData()

(x1, y1) = dataload.load_data("group0")
(x2, y2) = dataload.load_data("group1")
(x3, y3) = dataload.load_data("group2")
(x4, y4) = dataload.load_data("group3")
(x5, y5) = dataload.load_data("group4")
x = []
y = []
frame = dataload.get_nFrame()
cate = dataload.get_Cate()

x.append(x1.reshape(-1, 1, frame, dataload.m_size))
x.append(x2.reshape(-1, 1, frame, dataload.m_size))
x.append(x3.reshape(-1, 1, frame, dataload.m_size))
x.append(x4.reshape(-1, 1, frame, dataload.m_size))
x.append(x5.reshape(-1, 1, frame, dataload.m_size))
y.append(np_utils.to_categorical(y1, num_classes=cate))
y.append(np_utils.to_categorical(y2, num_classes=cate))
y.append(np_utils.to_categorical(y3, num_classes=cate))
y.append(np_utils.to_categorical(y4, num_classes=cate))
y.append(np_utils.to_categorical(y5, num_classes=cate))

f = open("5fold_test2.txt", "w")

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
        batch_input_shape=(None, 1, frame, dataload.m_size),
        filters=dataload.m_size,
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

    model.fit(X_train, y_train, epochs=10)

    print('\nTesting ------------')

    loss, accuracy = model.evaluate(X_test, y_test)

    print('\ntest loss: ', loss)
    print('\ntest accuracy: ', accuracy)
    f.write("\nRound" + str(i))
    f.write('\ntest loss: ' + str(loss))
    f.write('\ntest accuracy: '+ str(accuracy))

f.close()




