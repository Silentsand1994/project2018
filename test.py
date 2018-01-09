from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import RMSprop
import input

(X_train, y_train, frame, cate) = input.load_data("wav")
(X_test, y_test, frame, cate) = input.load_data("wav")

y_train = np_utils.to_categorical(y_train, num_classes=cate)
y_test = np_utils.to_categorical(y_test, num_classes=cate)


model = Sequential([
    Dense(32, input_dim=frame*14),
    Activation('relu'),
    Dense(cate),
    Activation('softmax'),
])

# Another way to define your optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# We add metrics to get more results you want to see
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print('Training ------------')

model.fit(X_train, y_train, epochs=10, batch_size=32)

print('\nTesting ------------')

loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)