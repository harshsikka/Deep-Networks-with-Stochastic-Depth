# Deep Networks with Stochastic Depth

Status:
resnet blocks are generating and passing
training is working
next steps: 
complete training process
process should dynamically create graphs based on probability 
and train them
- probably try to do this within a session
- review details of convolutional architecture in resnet

Overall Approach:

model res blocks and other layers
store res block for later access

setup selection probability function
    outputs the prob for each layer

setup training function
    training function will access res block store and will connect resblocks
    the probability for the given res block will be determined by the probability function, and will be factored into the connection

setup testing function
    will use updated test rule to account for all connections being active

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

The authors propose a 