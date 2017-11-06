from __future__ import print_function
import tensorflow as tf
import numpy as np 
import csv 
from sklearn.model_selection import train_test_split
from random import randint


std_deviation = np.ones(13) 
std_deviation=[2.19,1.74,1.38,1.10,0.875,0.69,0.55,0.43,0.34,0.27,0.21,0.17,0.13]


def gaussian_noise_layer(input_layer,std):
    noise=tf.random_normal(shape=tf.shape(input_layer),mean=0.0,stddev=std,dtype=tf.float32)
    return input_layer + noise  

# Parameters
learning_rate = 0.0001 
num_steps = 2560 ## number of iterations for training 
batch_size = 256 ## batch size for stochastic gradient descent method 
display_step = 256

# Network Parameters
n_hidden_1 = 16 # 1st layer number of neurons 2^4 
n_hidden_2 = 128 # 2nd layer number of neurons 2^7
n_normalization = 128 # 3rd layer number of neurons 
n_noise = 128 # 3rd layer number of neurons 
n_hidden_3 = 16
n_hidden_4 = 16

one_hot_vector = 16 
num_input = 16 

# tf Graph input
X = tf.placeholder("float", [None, one_hot_vector])
Y = tf.placeholder("float", [None, one_hot_vector])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3':tf.Variable(tf.random_normal([n_noise,n_hidden_4])),
    'out':tf.Variable(tf.random_normal([n_hidden_3,n_hidden_4]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3':tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_4]))
}


# Create model
for i in range(0,13):
    def neural_net(x):
        layer_1 = tf.nn.relu(tf.matmul(x, weights['h1']) + biases['b1'])
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']),biases['b2'])
        layer_3 = tf.nn.l2_normalize(layer_2,1, epsilon=1e-12, name=None)
        layer_4 = gaussian_noise_layer(layer_3,std_deviation[i]) 
        layer_5 = tf.nn.relu(tf.matmul(layer_4,weights['h3'])+biases['b3'])
        out_layer = tf.nn.softmax(tf.matmul(layer_5,weights['out'])+biases['out'])
        return out_layer

    # Construct model
    logits = neural_net(X)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=X))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(X, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        for step in range(1, num_steps+1):
            identityMatrix = np.eye(16)  
            batch_x = np.ones((batch_size,16))
            messages = [randint(1,16) for p in range(0,batch_size)]
            for i in range (0,batch_size): 
                batch_x[i] = identityMatrix[messages[i]-1]

            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_x})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_x})  
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Bit Error Rate= " + \
                      "{:.6f}".format(1-acc))

        print("Optimization Finished!")

        identityMatrix = np.eye(16)  
        batc_x = np.ones((100000,16))
        messages = [randint(1,16) for p in range(0,100000)]
        for i in range (0,100000): 
            batc_x[i] = identityMatrix[messages[i]-1]


        # Calculate accuracy for MNIST test images
        print("Testing accuracy:", \
            sess.run(accuracy, feed_dict={X: batc_x,
                                          Y: batc_x}))    