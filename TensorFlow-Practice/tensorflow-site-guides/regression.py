# using tensorflow for regression analysis on the standard automotive 
# miles-per-galon dataset

#import path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns # pretty graph library

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow version: ", tf.__version__)
# load the dataset from an external location

dataset_path = keras.utils.get_file("auto-mpg.data",
              "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

# import the data into a pandas dataframe
column_names = ["MPG", "Cylinders", "Displacement", "Horsepower", 
                "Weight", "Acceleration", "Model year", "Origin"]
raw_dataset = pd.read_csv(dataset_path, 
                          names=column_names,
                          na_values="?",
                          comment="\t",
                          sep= " ",
                          skipinitialspace=True)

dataset = raw_dataset.copy()

# take a look at a few examples
print(dataset.tail())

# there are a few empty, or NaN values in the dataset. Let's see how many
print("number of missing datapoints:\n", dataset.isna().sum())

# for simplicity we'll just drop the incomplete rows, to be more thorough we could estimate them
dataset = dataset.dropna()

# the "Origin" column is really a categorical label, but we want a unique value for each category
# so pop and add a new column of 1 or 0 values for each category
origin = dataset.pop("Origin")

# add a new column based on the values in "Origin"
dataset["USA"] = (origin == 1) * 1
dataset["Europe"] = (origin ==2 ) * 1
dataset["Japan"] = (origin == 3) * 1

# now we're ready to split into training and test data
training_data = dataset.sample(frac=0.8, random_state=0)
testing_data = dataset.drop(training_data.index)

# take a look at some of the data with seaborn to get a feel for it
sns.pairplot(training_data[["MPG","Cylinders", "Displacement", "Weight"]], diag_kind="kde")

# take a look at the DataFrame values too
training_stats = training_data.describe()
training_stats.pop("MPG")
training_stats = training_stats.transpose()
print("basic stats on our training set:")
print(training_stats)

# split the training set into features and value we want to predict
training_labels = training_data.pop("MPG") # our target to learn to predict
testing_labels = testing_data.pop("MPG")

# the means and standard deviations in our feature set are all over the place
# normalize them to mean = 0, stddev = 1
def normalize(x):
  return (x - training_stats["mean"]) / training_stats["std"]

normalized_training_data = normalize(training_data)
normalized_testing_data = normalize(testing_data)

# we're now ready to build our tensorflow model.
# it will be a sequential 2 layer model that's fully connected (dense)
model = keras.models.Sequential([
                  layers.Dense(64, activation=tf.nn.relu, input_shape=[len(training_data.keys())]),
                  layers.Dense(64, activation=tf.nn.relu),
                  layers.Dense(1)]) # our single regression value output

optimizer = tf.keras.optimizers.RMSprop(0.001) # RMS error, 0.001 training rate

model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['mean_absolute_error', 'mean_squared_error'])

model.build()

# we can look at the model summary stats
print(model.summary())

# our model is built, but not trained yet.
# Let's use with some of our training data to see our things are connected
example_batch = normalized_training_data[:10]
example_result = model.predict(example_batch)
print(example_result) # basically random numbers from the initialized weights

# training the model
print("Training model. This will take a moment")
num_epochs = 1000
model_history = model.fit(normalized_training_data.values,
                    training_labels.values,
                    epochs=num_epochs,
                    verbose=0,
                    validation_split=0.2)

# we stored the model training process in our 'model_history' object to view afterward
hist = pd.DataFrame(model_history.history)
hist['epoch'] = model_history.epoch
print(hist.tail())

# plot stuff to make it easier to understand
plt.figure()
plt.title("Traning and Validation Mean Absolute Error over Epochs")
plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error [MPG]')
plt.plot(hist['epoch'], hist['mean_absolute_error'],
         label='Train Error')
plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
         label = 'Val Error')
plt.ylim([0,5])
plt.legend()

plt.figure()
plt.title("Traning and Validation Mean Square Error over Epochs")
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error [$MPG^2$]')
plt.plot(hist['epoch'], hist['mean_squared_error'],
         label='Train Error')
plt.plot(hist['epoch'], hist['val_mean_squared_error'],
         label = 'Val Error')
plt.ylim([0,20])
plt.legend()
#plt.show()

# from the two training graphs we see that validation error levels off pretty soon
# we can stop the training process early by monitoring the validation error

early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=10)
# patience parameter is the number of epochs to wait before calling early stopping
# this to make sure we don't stop on a single noisy blip 

print("Re training model with early stopping. This will take a moment")
# need to build a new model object
# calling .fit() method starts from current model state which is already trained
model_2= keras.models.Sequential([
                  layers.Dense(64, activation=tf.nn.relu, input_shape=[len(training_data.keys())]),
                  layers.Dense(64, activation=tf.nn.relu),
                  layers.Dense(1)]) # our single regression value output

optimizer = tf.keras.optimizers.RMSprop(0.001) # RMS error, 0.001 training rate

model_2.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['mean_absolute_error', 'mean_squared_error'])

model_2.build()
model_history_2 = model_2.fit(normalized_training_data.values,
                          training_labels.values,
                          epochs=num_epochs,
                          validation_split=0.2,
                          verbose=1,
                          callbacks=[early_stop])

# replot curves to see our improvement with early stopping
hist_2 = pd.DataFrame(model_history_2.history)
hist_2['epoch'] = model_history_2.epoch

plt.figure()
plt.title("Traning and Validation Mean Absolute Error over Epochs\n - early stopping")
plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error [MPG]')
plt.plot(hist_2['epoch'], hist_2['mean_absolute_error'],
         label='Train Error')
plt.plot(hist_2['epoch'], hist_2['val_mean_absolute_error'],
         label = 'Val Error')
plt.ylim([0,5])
plt.legend()

plt.figure()
plt.title("Traning and Validation Mean Square Error over Epochs\n - early stopping ")
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error [$MPG^2$]')
plt.plot(hist_2['epoch'], hist_2['mean_squared_error'],
         label='Train Error')
plt.plot(hist_2['epoch'], hist_2['val_mean_squared_error'],
         label = 'Val Error')
plt.ylim([0,20])
plt.legend()
#plt.show()

# with our efficiently trained model we are ready to make predictions with it
loss, mean_abs_error, mean_square_error = model_2.evaluate(normalized_testing_data.values, 
                                                          testing_labels.values,
                                                          verbose=0)

print("testing set mean absolute error:", mean_abs_error)
print("testing set mean square error:", mean_square_error)

test_predictions = model_2.predict(normalized_testing_data).flatten()

plt.figure()
plt.scatter(testing_labels, test_predictions)
plt.title("Testing predictions of our efficient model")
plt.xlabel("True predictions [MPG]")
plt.ylabel("Predictions[MPG]")
plt.axis("equal")
plt.axis("square")
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
plt.plot([-100, 100], [-100, 100])

# we can also look at our model error distribution
error = test_predictions - testing_labels
plt.figure()
plt.title("Error distribution of our MPG regression model")
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")
plt.show()

