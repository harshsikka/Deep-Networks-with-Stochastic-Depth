import tensorflow.compat.v1 as tf
import utilities


# Basic Building Block for Resnet Architecture

def batch_norm(inputs):
  return tf.layers.batch_normalization(inputs=inputs, axis=3,
            momentum=0.997, epsilon=1e-5, center=True, scale=True, fused=True)

def convolution(inputs, filters, kernel_size, strides):
    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer())
    
def resBlock(inputs, filters, kernel_size, strides):

        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(tensor=inputs, paddings=[[0, 0], 
            [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        
        x = convolution(padded_inputs, filters, kernel_size, strides)

        x = batch_norm(x)
        
        x = tf.nn.relu(x)

        x = convolution(x, filters, kernel_size, 1)
        
        outputs = batch_norm(x)

        return outputs

    
def inputLayers(inputs, filters, kernel_size, strides):

        x = convolution(inputs, filters, kernel_size, strides)

        x = batch_norm(x)
        
        outputs = tf.nn.relu(x)

        return outputs

def outputLayers(inputs):
        
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        
        logits = tf.layers.dense(x,10)
        
        outputs = logits
        
        return outputs
          