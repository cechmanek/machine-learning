'''
convolutional network for digit classification

follows the example starting on page 120
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# load and normalize data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1).astype('float32') / 255
x_test = x_test.reshape(10000,28,28,1).astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# define and build model
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

model.summary()

# now we can fit to our data
model.fit(x_train, y_train, epochs=5, batch_size=64)


# lets evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('test loss is ', test_loss)
print('test accuracy is ', test_accuracy)
