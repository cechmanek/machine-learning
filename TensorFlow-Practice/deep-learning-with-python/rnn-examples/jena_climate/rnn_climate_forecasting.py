'''
time series forecasting of jena climate data
download the dataset via:
wget https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip
unzip https....
'''


import os
import numpy as np
import matplotlib.pyplot as plt

# our dataset takes readings of temp, pressure, humidity, etc every 10 minutes
file_name = 'jena_climate_2009_2016.csv'

f = open(file_name)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print("data consists of :",header)
print("there are {} datapoints".format(len(lines)))

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    float_data[i,:] = [float(x) for x in line.split(',')[1:]]

# look at some of the data from our dataset
temp = float_data[:,1] # temperature in degrees celcius
plt.plot(range(len(temp)), temp)
plt.title('temperature in degrees C')

plt.figure()
plt.plot(range(1440), temp[:1440])
plt.title('temperature in degrees C for first 10 days (1440 data points)')
#plt.show()
# the data shows both annual and daily periodicity


# time to preprocess the data for deep learning, normalize and standardize
mean = float_data[:200000].mean(axis=0) # mean of all columns
float_data -= mean

std = float_data[:200000].std(axis=0)
float_data /= std

# make a generator for getting our sample batches
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    '''
    data: original array of floating poitn data which is already normalized
    lookback: how many timesetsp back the input should go
    delay: how man ytimesteps in the future the target should be
    min_/max_index: indices in data array that delimit timesteps to draw from. used for splitting
    shuffle: whether to shuffle samples or draw in chronological order
    batch_size: number of samples per batch
    step: the period. raw data has a period of 10 minutes. step of 6 down samples to once per hour
    '''
    if max_index is None:
        max_index = len(data) - delay -1

    i = min_index + lookback
    while True:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback 
            rows = np.arange(i, min(i+ batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))

        targets = np.zeros((len(rows)))

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        
        yield samples, targets


# now use this generator function to create a train, validation and test set

lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=0,
                        max_index=200000,
                        shuffle=True,
                        step=step,
                        batch_size=batch_size)

validation_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=200001,
                        max_index=300000,
                        shuffle=False,
                        step=step,
                        batch_size=batch_size)

test_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=300000,
                        max_index=None, # go to end of dataset
                        shuffle=False,
                        step=step,
                        batch_size=batch_size)

val_steps = 300000 - 200001 - lookback # steps to draw from val_gen to see entire validation set
test_step = len(float_data) - 300001 - lookback # steps to see entire test set


# before diving into complex deep ML models lets build a naive baseline approach:
# assume a 2h hour periodicity. so predict that the temperature 24 hours from now == temp_now

# will use mean absolute error (MAE) as our metric:
# mae =  np.mean(np.abs(preds - targets))

def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        if step%100 == 0:
            print('on step {} of {}'.format(step, val_steps))
        samples, targets = next(validation_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)

    print('our naive 24hr period method has an mae score of: ', np.mean(batch_maes))

print('evaluating naive method ...')
#evaluate_naive_method() # VERY slow (25min). gives a MAE of about 0.29, or +-2.57 C
print('finished evaluating naive method.')


# the next step is to build a simple dense network that accepts flattened data input

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GRU, Bidirectional
from tensorflow.keras.optimizers import RMSprop

print("training standard forward net")
model = Sequential()
model.add(Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(Dense(32, activation='relu'))
model.add(Dense(1)) #linear output for regression problem. we're only predicting temp celcius

model.compile(loss='mae', optimizer=RMSprop())

history = model.fit(train_gen, steps_per_epoch=500, epochs=20, validation_data=validation_gen, validation_steps=val_steps//128)

# time to view our model results 

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) +1 )

plt.figure()
plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b-', label='validation loss')
plt.title('Forward Net training and validation losses')
plt.legend()
#plt.show()


# the next step is to build a recurrent model to see if it out performs a regular forward net

print("training shallow recurrent net")
model = Sequential()
model.add(GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(Dense(1))

model.compile(loss='mae', optimizer=RMSprop())

history = model.fit(train_gen, steps_per_epoch=500, epochs=20, validation_data=validation_gen, validation_steps=val_steps//128)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) +1 )

plt.figure()
plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b-', label='validation loss')
plt.title('Recurrent Net training and validation losses')
plt.legend()
#plt.show()

# the GRU recurrent network gives a best-case MAE of about 0.265, or +-2.35 C

# to combat the overfitting we're seeing we want to use droput, but dropout for RNNS is special
# we can't intermingle dropout layers, the droput has to be built into each GRU or LSTM
# RNN units have two types of dropout, a mask for the input, and a mask for the recurrent state

print("training shallow recurrent net with dropout")
model = Sequential()
model = Sequential()
model.add(GRU(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, float_data.shape[-1])))
model.add(Dense(1))

model.compile(loss='mae', optimizer=RMSprop())

# train for twice as many epochs since we added dropout
history = model.fit(train_gen, steps_per_epoch=500, epochs=40, validation_data=validation_gen, validation_steps=val_steps//128)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) +1 )

plt.figure()
plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b-', label='validation loss')
plt.title('Recurrent Net training and validation losses')
plt.legend()
#plt.show()

# our next improvement is to use a deep recurrent net
print("training deep recurrent net with dropout")
model = Sequential()
model.add(GRU(32,
             dropout=0.1,
             recurrent_dropout=0.5, 
             input_shape=(None, float_data.shape[-1]),
             return_sequences=True)) # need to return full sequence for intermediate layers
model.add(GRU(64, dropout=0.1, recurrent_dropout=0.5 ))
model.add(Dense(1))

model.compile(loss='mae', optimizer=RMSprop())

# train for twice as many epochs since we added dropout
history = model.fit(train_gen, steps_per_epoch=500, epochs=40, validation_data=validation_gen, validation_steps=val_steps//128)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) +1 )

plt.figure()
plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b-', label='validation loss')
plt.title('Deep Recurrent Net training and validation losses')
plt.legend()
#plt.show()

# our last improvement is to use a bidirectional recurrent net
print("training shallow bidirectional recurrent net with dropout")
model = Sequential()
model.add(Bidirectional(GRU(32,
                            dropout=0.1,
                            recurrent_dropout=0.5, 
                            input_shape=(None, float_data.shape[-1]),
                            return_sequences=True) # return full sequence for intermediate layers
                       )
         )
model.add(Dense(1))

model.compile(loss='mae', optimizer=RMSprop())

# train for twice as many epochs since we added dropout
history = model.fit(train_gen, steps_per_epoch=500, epochs=40, validation_data=validation_gen, validation_steps=val_steps//128)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) +1 )

plt.figure()
plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b-', label='validation loss')
plt.title('Bidirectional Recurrent Net training and validation losses')
plt.legend()
plt.show()


