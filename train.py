import tensorflow.compat.v1 as tf
from tensorflow.keras import utils, datasets, layers, models
import matplotlib.pyplot as plt
import model
import numpy as np

tf.disable_eager_execution()

# Cifar-10 Dataset Setup
size = 32
num_classes = 10

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images = train_images / 255
test_images = test_images / 255

# Convert class vectors to binary class matrices.
y_train = utils.to_categorical(train_labels, num_classes)
y_test = utils.to_categorical(test_labels, num_classes)

print(y_test.shape)



# calculate p_l from paper, the probability decay that determines the bernouli rv. Equation (4)
def calc_p_l(l, L, p_L):
    decay = (l / L) * (1 - p_L)
    p_l = 1 - decay
    return p_l

# calculate b_l from paper, determining activation of current ResBlock
def calc_b_l(l, L, p_L):
    p_l = calc_p_l(l, L, p_L)
    return np.random.choice([1, 0], p=[p_l, 1 - p_l]) 

# storage for blocks
block_storage = []

# input and output block definitions
input_layers = tf.make_template('input_layers', model.inputLayers)
output_layers = tf.make_template('output_layers', model.outputLayers)

# initialize parameter sharing and store blocks
for i in range(0,3):
    for j in range(0,18): 
        block_storage.append(tf.make_template('group_'+str(i)+'_block_'+str(j), model.resBlock))

# function defining learning rule in equation (2) from paper
def updateSubNetworkBlocks(start, end, input, filter, subgraph_length):
    x = input
    for i in range(start,end): 
        b_l = calc_b_l(i,54,0.5)
        if (b_l == 1):
            subgraph_length += 1

        x = tf.nn.relu(b_l * block_storage[i](x, filter, 1, 1) + x)
    return x, subgraph_length

# function defining evaluation rule in equation (6) from paper
def updateFullNetworkBlocks(start, end, input, filter):
    x = input
    for i in range(start,end): 
        p_l = calc_p_l(i,54,0.5)
        x = tf.nn.relu(p_l * block_storage[i](x, filter, 1, 1) + x)
    return x

# function to define subgraph creation and corresponding minibatch training
def trainNetwork(block_storage, batch_images, batch_labels):
    subgraph_length = 3
    data = tf.placeholder(tf.float32, [None, size, size, 3])
    label = tf.placeholder(tf.float32, [None,10])

    x = input_layers(data, 16, 3, 2)
    x = block_storage[0](x, 16, 1, 1)
    x, subgraph_length = updateSubNetworkBlocks( 1, 18, x, 16, subgraph_length)
    x = tf.nn.relu(block_storage[18](x, 32, 1, 1))
    x, subgraph_length = updateSubNetworkBlocks( 19, 36, x, 32, subgraph_length)
    x = tf.nn.relu(block_storage[36](x, 64, 1, 1))
    x, subgraph_length = updateSubNetworkBlocks( 37, 54, x, 64, subgraph_length)
    x = output_layers(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits=x))

    optimizer = tf.train.MomentumOptimizer(.1, .9, use_nesterov=True).minimize(loss)
    tf.global_variables_initializer().run()
    
    sess.run(optimizer, feed_dict={
        data: batch_images, label: batch_labels,
        })

    # subgraph length aligns with length expectations in eq (5)
    print('number of activated blocks in training: ' + str(subgraph_length)) 

# function to evaluate over the whole network
def testNetwork(block_storage, batch_images, batch_labels):
    data = tf.placeholder(tf.float32, [None, size, size, 3])
    label = tf.placeholder(tf.float32, [None,10])

    x = input_layers(data, 16, 3, 2)
    x = block_storage[0](x, 16, 1, 1)
    x = updateFullNetworkBlocks( 1, 18, x, 16)
    x = tf.nn.relu(block_storage[18](x, 32, 1, 1))
    x = updateFullNetworkBlocks( 19, 36, x, 32)
    x = tf.nn.relu(block_storage[36](x, 64, 1, 1))
    x = updateFullNetworkBlocks( 37, 54, x, 64)
    x = output_layers(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits=x))

    tf.global_variables_initializer().run()
    loss_val =  loss.eval(feed_dict={
        data: batch_images, label: batch_labels,
    })

    correct_pred = tf.equal(tf.argmax(x, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    acc_val =  accuracy.eval(feed_dict={
        data: batch_images, label: batch_labels,
    })

    print('Loss: ', loss_val)
    print('Accuracy: ', acc_val)

#session
with tf.Session() as sess:
    # setup train, val split
    train_images = train_images[:45000]
    val_images = train_images[45000:50000]

    train_labels = train_labels[:45000]
    val_labels = train_labels[45000:50000]

    #setup minibatches from paper
    num_batches = len(train_images) // 128

    # network initiliazation is happening directly within the session to dynamically create subgraphs for each minibatch
    for epoch in range(500):
        for i in range(num_batches):
            print('Training Epoch: ' + str(epoch) + ' Minibatch: ' + str(i))
            trainNetwork(block_storage, train_images[128*i: 128 + 128*i], y_train[128*i: 128 + 128*i])
            print('Testing Epoch: ' + str(epoch) + ' Minibatch: ' + str(i))
            testNetwork(block_storage, train_images[128*i: 128 + 128*i], y_train[128*i: 128 + 128*i])
            


