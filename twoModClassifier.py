from __future__ import print_function
import tensorflow as tf
import numpy as np 
import csv 
from sklearn.model_selection import train_test_split

filename = 'myData_SNR20.csv'

rawdata = open(filename,'rt')

data=np.loadtxt(rawdata,delimiter=",")


zeros = np.zeros((20000,1),dtype=np.int)
ones = np.ones((20000,1),dtype=np.int)
twos = ones + 1 
fours = zeros + 4 

qpsk = zip(data[:,4],data[:,5],fours[:,0])
qam = zip(data[:,2],data[:,3],twos[:,0])
bpsk = zip(data[:,0],data[:,1],ones[:,0])

final_data = bpsk + qam + qpsk 

import random 
random.shuffle(final_data)

train_data = final_data[:int(len(final_data)*0.8)]  
test_data = final_data[int(len(final_data)*0.8):]

list1_train_I, list2_train_Q,list3_train_label = zip(*train_data)
list1_test_I, list2_test_Q,list3_test_label = zip(*test_data)

# Parameters
learning_rate = 0.001 
num_steps = 2000 ## number of iterations for training 
batch_size = 500 ## batch size for stochastic gradient descent method 
display_step = 200

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_hidden_3 = 256 # 3rd layer number of neurons 
n_hidden_4 = 256 # 3rd layer number of neurons 
num_input = 2 # Should be 2 for our case I and Q component 
num_classes = 3 # Should be 2 for our case 4QAM or 16QAM

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3])),
    'h4':tf.Variable(tf.random_normal([n_hidden_3,n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_hidden_4, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3':tf.Variable(tf.random_normal([n_hidden_3])),
    'b4':tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.relu(tf.matmul(x, weights['h1']) + biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.relu(tf.matmul(layer_1, weights['h2']) + biases['b2'])
    # Layer 3 hidden fully connected 
    layer_3 = tf.nn.relu(tf.matmul(layer_2,weights['h3'])+biases['b3'])
    layer_4 = tf.nn.relu(tf.matmul(layer_3,weights['h4'])+biases['b4'])
    #Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        p = np.random.permutation(len(list1_train_I))
        p = p - 1 

        batch_x = np.ones((batch_size,num_input))
        batch_y = np.zeros((batch_size,num_classes))

        for i in range (0,batch_size):
            batch_x[i][0] = list1_train_I[p[i]]
            batch_x[i][1] = list2_train_Q[p[i]]
            batch_y[i][0] =  int(list3_train_label[p[i]]==1) 
            batch_y[i][1] =  int(list3_train_label[p[i]]==2)
            batch_y[i][2] =  int(list3_train_label[p[i]]==4)

        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    batc_x = np.ones((int(len(list1_test_I)),num_input))
    batc_y = np.ones((int(len(list1_test_I)),num_classes))

    for i in range(0,(int(len(list1_test_I)))):
        batc_x[i][0]=list1_test_I[i]
        batc_x[i][1]=list2_test_Q[i]
        batc_y[i][0] =  int(list3_test_label[i]==1) 
        batc_y[i][1] =  int(list3_test_label[i]==2)
        batc_y[i][2] =  int(list3_test_label[i]==4)


    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: batc_x,
                                      Y: batc_y}))