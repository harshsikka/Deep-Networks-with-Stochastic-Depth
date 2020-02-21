# Deep Networks with Stochastic Depth

The following is an implementation of *Deep Networks with Stochastic Depth* by Huang et al. 

## Paper Summary
The authors propose a dynamic layer skipping scheme for deep neural networks that reduced training time significantly but
allows for competitive performance on benchmark tasks.

### Background and Research Problem
They review 3 core issues with training very deep neural networks:

1. Vanishing gradients: as gradients-based adjustments are propagated through the layers of a network, smaller weight magnitudes
   lead to lesser impact on earlier weights. Skip connections and batchnorm help this.

2. Diminishing reuse of features: representations from earlier layers don't inform later ones for similar reasons to the vanishing
   gradient issue. Resnets aim to fix this through identity skip connections.

3. Long training time: training of deeper networks is intuitively more time and compute intensive.

One of the key issues in Deep Learning, both at the time of writing this paper and today, is that small networks are not expressive
enough to tackle complex problems, but large ones face serious learning problems (see above) through the current training schemes.

### Proposed Solution

The authors propose a mechanism to skip layer updates stochastically, through a decaying probability. At training, layers may or may
not get updated, so computation is saved. At testing, all layers are involved, allowing for utilization of the whole network. Another 
perspective on this is that a given resnet becomes an ensemble of many smaller resnets that are basically graphs with skipped portions.
The results are more than competitive. 

## Implementation Details

### Overview
This implementation makes use of Tensorflow v1. The ResNet is built of the same repeated motif with different filter sizes, so a basic resnet
building block is implemented and parameters are shared using tf.maketemplate to allow for subsequent updates. 

During training, blocks are connected in the larger supergraph using the stochastic mechanism introduced in the paper. This leads to various
subgraphs being generated and trained. For evaluation, the all the blocks are connected through the forward pass rule introduced in the paper, 
and evaluated.
