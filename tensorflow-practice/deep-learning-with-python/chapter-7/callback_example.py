'''
implements a custom Keras callback class that can be passed to model.fit() method
this follows the code example starting on page 251

build a class that extends keras.callbacks.Callback
implement the methods:
on_epoch_begin
on_epoch_end
on_batch_begin
on_batch_end
on_train_begin
on_train_end

this class has access to self.model and self.validation_data
each method above is called with the Logs dict, which contains info about the last epoch
'''

import tensorflow.keras as keras
import numpy as np

# define a class that logs the activation output of each layer after each epoch
class ActivationLogger(keras.callbacks.Callback):
    
    def set_model(self, model):
        self.model = model
        layer_outputs = [layer.output for layer in model.layers]
        self.activations_model = keras.models.Model(model.input, layer_outputs)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation-data is None:
            raise RuntimeError('Requires validation_data.')
        validation_sample = self.validation_data[0][0:1]
        activations = self.activations_model.predict(validation_sample)
        f = open('activations_at_epoch_' + str(epoch) + '.npz', 'w') # open np array file to write
        np.savez(f, activations) # as numpy array file, with extension .npz
        f.close()
