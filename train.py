import tensorflow.compat.v1 as tf
from tensorflow.keras import utils, datasets, layers, models
import matplotlib.pyplot as plt
import model
import numpy as np

tf.disable_eager_execution()

# Cifar-10 Dataset Setup
size = 32

strides = [1, 2, 2]
filter_sizes = [16, 32, 64]

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images = train_images / 255
test_images = test_images / 255
y_train = utils.to_categorical(train_labels, 10)
y_test = utils.to_categorical(test_labels, 10)

# Utility Functions

def p_l(l, L, p_L):
    decay = (l / L) * (1 - p_L)
    return 1 - decay

# storage for blocks
block_storage = []

# input layer definition
input_layers = tf.make_template('input_layers', model.inputLayers)
output_layers = tf.make_template('output_layers', model.outputLayers)

# output layer definition

for i in range(0,3):
    for j in range(0,18): 
        block_storage.append(tf.make_template('group_'+str(i)+'_block_'+str(j), model.resBlock))
        
        # add logic about being first of the 2nd and 3rd group here

# initialize network
data = tf.placeholder(tf.float32, [None, size, size, 3])
label = tf.placeholder(tf.float32, [None,1])

x = input_layers(data, 16, 3, 1)

for i in range(0,17): 
    x = block_storage[i](x, 16, 1, 2)

for i in range(17,36): 
    x = block_storage[i](x, 32, 1, 2)

for i in range(36,54): 
    x = block_storage[i](x, 64, 1, 2)

x = output_layers(x)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits=x))

optimizer = tf.train.AdamOptimizer(.1).minimize(loss)

#session

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(1,101):
        prev_idx = i - 1
        
        sess.run(optimizer, feed_dict={
          data: train_images[prev_idx*100:i*100], label: train_labels[prev_idx*100:i*100]
        })

    loss_test = loss.eval(feed_dict={
          data: train_images[1:3], label: train_labels[1:3]
        })

print(len(block_storage)) 
print(train_images.shape)
print(test_labels[0])
print(loss_test)
