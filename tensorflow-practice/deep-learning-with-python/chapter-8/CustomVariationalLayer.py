'''
custom class extending keras layer class used in variational auto encoder example
Chollet is a garbage programmer so his example includes this class definition inline in a script
'''
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as backend
import tensorflow.keras as keras

class CustomVariationalLayer(Layer):
    
    def vae_loss(self, x, z_decoded, z_mean, z_log_var):
        x = backend.flatten(x)
        z_decoded = backend.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        k1_loss = -5e-4 * backend.mean(1 + z_log_var - backend.square(z_mean) - backend.exp(z_log_var), axis=-1)
        # z_mean and z_log_var are globals. Chollet you eat shit again
        return backend.mean(xent_loss + k1_loss)

    def call(self, inputs): # must implement call for custom layers
        x = inputs[0]
        z_decoded = inputs[1]
        z_mean = inputs[2]
        z_log_var = inputs[3]
        loss = self.vae_loss(x, z_decoded, z_mean, z_log_var)
        self.add_loss(loss, inputs=inputs)
        return x # not needed, but call() needs to return something
