# using TensorFlow instead of Torch in a neural network equivalent to warm_up_numpy.py example
# NOTE: this is TensorFlow V1 and not compatible with TensorFlow V2
# There have been substantial breaking changes between V1 and V2
import tensorflow as tf
import numpy as np

N = 64 # batch size
D_in = 1000 # input dimension
H = 100 # hidden layer size
D_out = 10 # output dimension

# create placeholders that will be filled with data once we run the graph
x = tf.placeholder(tf.float32, shape=(None, D_in))
y = tf.placeholder(tf.float32, shape=(None, D_out))

# create variables for the weights and initialize them
# Variables persist their values across executions of the graph
w1 = tf.Variable(tf.random_normal((None, D_in)))
w2 = tf.Variable(tf.random_normal((None, D_in)))


# define the forward pass. This doesn't execute code, it only builds the graph
h = tf.matmul(x, w1)
h_relu = tf.maximum(h , tf.zeros(1))
y_pred = tf.matmul(h_relu, w2)

# define the loss
loss = tf.reduce_sum((y - y_pred)**2.0)

# define the gradients w.r.t. the loss
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

#define the learning rate and update steps
learning_rate = 1e-6
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)

# above we have been defining the graph, but have not executed anyting yet
# now we will exectute code through our graph
with tf.Session() as sess:
  # run through the graph once to initialize our Variables w1 and w2
  sess.run(tf.global_variables_initializer())

  # create numpy arrays to hold the actual data for x and y
  x_value = np.random.randn(N, D_in)
  y_value = np.random.randn(N, D_out)

  # finally, we iterate over our graph
  for t in range(500):
    # Execute the graph many times. Each time it executes we want to bind x_value to x and y_value
    #  to y, specified with the feed_dict argument.
    # Each time we execute the graph we want to compute the values for loss, # new_w1, and new_w2;
    # the values of these Tensors are returned as numpy arrays.

    loss_value, _, _ = sess.run([loss, new_w1, new_w2], feed_dict={x:x_value, y:y_value})
    print('loss at step {} is {}'.format(t, loss_value))
