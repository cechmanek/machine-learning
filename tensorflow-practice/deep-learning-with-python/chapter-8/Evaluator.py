'''
class to to handle gradient and loss values while doing gradient ascent in style transfer
'''
import numpy as np
 
class Evaluator(object):
    def __init__(self, img_height, img_width, loss_and_grads_function):
        self.img_height = img_height
        self.img_width = img_width
        self.loss_value = None
        self.grad_values = None
        self.loss_and_grads = loss_and_grads_function

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, self.img_height, self.img_width, 3))
        # fetch_loss_ function defined in main script because Chollet has no fucking business 
        # calling himself a software developer, computer scientist, or engineer
        #outs = fetch_loss_and_grads([x]) 
        outs = self.loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
