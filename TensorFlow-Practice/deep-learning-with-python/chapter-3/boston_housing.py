'''
regression example

following code starting at page 85
'''

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# our data has multiple different scale. Normalize each feature to a normal distribution
mean = np.mean(train_data, axis=0)
stddev = np.std(train_data, axis=0)

x_train = (train_data - mean) / stddev
x_test = (test_data - mean) / stddev


# we don't have much data, so rather than split into train/validation, we'll do k-fold validation
# this requires that we train k identical models
def build_model():
    model = Sequential()

    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1)) # linear output for regression

    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
    return model

k = 4
num_val_samples =  len(train_data) // k
num_epochs = 100
all_scores = []
all_mae_histories = []

for i in range(k):
    print('processing on fold: ', i)
    
    # take the a slice section of data as validation
    validation_data = x_train[i*num_val_samples: (i+1) * num_val_samples]
    validation_target = train_labels[i*num_val_samples: (i+1) * num_val_samples]

    # combine the rest of the data that wasn't sliced out as validation data into training data
    train_data = np.concatenate([x_train[:i*num_val_samples], x_train[(i+1)*num_val_samples:]], axis=0)
    train_target = np.concatenate([train_labels[:i*num_val_samples], train_labels[(i+1)*num_val_samples:]], axis=0)

    model = build_model()
    history =  model.fit(train_data, train_target,
            epochs=num_epochs, batch_size=1, verbose=0)

    validation_mse, validation_mae = model.evaluate(validation_data, validation_target, verbose=0)
    all_scores.append(validation_mae)
    
    mae_history = history.history['mae']
    all_mae_histories.append(mae_history)

print(all_scores)

# looking at all_scores is a good way to evaluate if your model architecture and num_epochs are good
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

epoch_range = range(1, len(average_mae_history)+1)

plt.plot(epoch_range, average_mae_history, label='mean absolute error')
plt.xlabel('epochs')
plt.ylabel('mae')
plt.show()

# once you're happy with a model architecture we can train on the full train_data with those settings
# to obtain our final model

