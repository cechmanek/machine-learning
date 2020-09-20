'''
multi-output models with keras functional api
following code starting on page 240
'''

from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model

# build a 3-output model to predict a person's age, gender and income based on social media posts
# one regression and two non-exclusive classes

vocab_size = 50000
num_income_groups = 10 # treat income as categorical, not continuous

# build out the base graph relations
posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, vocab_size)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts) # Conv1D for text, LSTM is overkill
x = layers.MaxPool1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPool1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPool1D()(x)
x = layers.Dense(128, activation='relu')(x)

# build out each output branch separately # each attaches directly to 'x' above
age_prediction = layers.Dense(1, activation='linear', name='age')(x) # name layers similar to Input
income_prediction = layers.Dense(num_income_groups, activation='softmax', name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])

# when compiling we must specify a loss for each output branch
model.compile(loss=['mse','categorical_crossentropy', 'binary_crossentropy'], optimizer='adam')
# OR pass a dict. this requires that output layers are named
loss_dict = {'age':'mse', 'income':'categorical_crossentropy', 'gender':'binary_crossentropy'}
model.compile(loss=loss_dict, optimizer='adam')

# right now all the losses are weighted equally,
# but we probably don't want this as different loss metrics have different ranges of values
# we may also want to preferentially weight the importance of certain outputs
model.compile(loss=['mse','categorical_crossentropy', 'binary_crossentropy'],
              loss_weights=[0.25, 1.0, 10.0],
              optimizer='adam')

# we can also use a dict for loss weights like:
weights_dict = {'age':0.25, 'income':1.0, 'gender':10.0}

''' skip training on random data, but if we wanted to it would look like this
# can train in one of two ways, by passing a list of outputs, or a dictionary with keys==names
model.fit(posts, [age_targets, income_targets, gender_targets], epochs=10)

# OR pass a dict again
targets_dict = {'age':age_targets, 'income':income_targets, 'gender':gender_targets}
model.fit(posts, targets_dict, epochs=10, batch_size=64)
'''
