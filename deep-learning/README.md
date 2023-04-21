In this project I've implemented three 
types of convolutional neural network 
for images classification on CIFAR10 dataset and evalutaed their performance.
* simple CNN with Conv2D, MaxPooling2D, BatchNormalization and Dropout layers
* CNN with ResNet-like architecture
* VGG16 with transfer learning

In addition to these models, I used several data 
augmentation techniques, including basic techniques 
such as cropping, rotation by multiples of 90 degrees, 
and adding noise, as well as a more complex technique 
called cutmix. Cutmix involves randomly cropping and pasting 
parts of different images to create a new training image, 
with corresponding labels that are a weighted average of the 
original labels.

The training process can also be customized using various 
parameters related to the training process (such as batch size 
and learning rate) and the network architectures (such as the 
number of filters and number of ResNet blocks).