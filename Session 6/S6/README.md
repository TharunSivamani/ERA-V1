# Part 1

# Backpropagation

Backpropagation is a widely used algorithm for training feedforward neural networks. It computes the gradient of the loss function with respect to the network weights. It is very efficient, rather than naively directly computing the gradient concerning each weight. This efficiency makes it possible to use gradient methods to train multi-layer networks and update weights to minimize loss; variants such as gradient descent or stochastic gradient descent are often used.

Backpropagation can be written as a function of the neural network. Backpropagation algorithms are a set of methods used to efficiently train artificial neural networks following a gradient descent approach which exploits the chain rule.

The basic idea behind backpropagation is to propagate the errors backward through the network, from the output layer to the hidden layers, and adjust the weights and biases of the network accordingly. This process is performed iteratively to minimize the difference between the predicted output and the actual output (i.e., reduce the loss).

<br>

## Neural Network
<br>

![simple_perceptron_model-1](../../Results/Session%206/nn.png)

<br>

## Weight Initialization
<br>

![image](../../Results/Session%206/nnweight.png)

## Forward pass
First we need to calculate the loss. For that we will use the input and following equations to forward pass the network.
<br>

![image](../../Results/Session%206/forwardpass.png)

<br>

## Calculating gradients wrt w5
The following equations are for calculating the gradients with respect to the weight w5
<br>

![image](../../Results/Session%206/weight5.png)

<br>

## Calculating gradients in layer 2
Based on the above equations we will write equations for w5, w6, w7, w8 at layer 2
<br>

![image](../../Results/Session%206/layer2.png)

<br>

## Calculating gradients intermediate step for layer 1
Now for calculating the gradients for layer 1, we need the gradients wrt h1 and h2
<br>

![image](../../Results/Session%206/gradientlayer1.png)

<br>

## Calculating gradients in layer 1
<br>

![image](../../Results/Session%206/layer10.png)

<br>

## Calculating gradients in layer 1
<br>
Using the above equation we can calculate the gradients for layer 1
<br>
<br>

![image](../../Results/Session%206/layer11.png)
<br>

Finally we will update the weights based on the calculated gradients. Based on the new weights, we will calculate the losses and do the above steps iteratively.
<br>


## Variation of Lossess wrt Learning Rate (Refer Excel Sheet-2)
<br>
We can clearly see that as the Learning Rate (in the range from 0.1 to 2) increases, the loss reduces at a faster rate.
<br>
<br>

![image](../../Results/Session%206/result.png)



## Backpropagation Algorithm
<br>
The backpropagation algorithm is used to train the neural network. It involves the following steps :   
<br>

1. `Forward propagation`: The input data is propagated forward through the network, and activations are computed for each layer.   
2. `Compute loss`: The difference between the predicted output and the actual output (label) is calculated.   
3. `Backward propagation`: Errors are propagated backward through the network to adjust the weights and biases using gradient descent optimization.   
4. `Update weights`: The weights and biases of the network are updated based on the calculated gradients.   
5. `Repeat` steps 1-4 for multiple epochs until the network converges. 


<br>

## Result Graphs
<br>

| Image                      | Learning Rate | Image                      | Learning Rate |
| :--------------------------: | :-------------: | :--------------------------: | :-------------: |
| ![Image 0.1](../../Results/Session%206/lr0.1.png)    |   LR=0.1      | ![Image 0.5](../../Results/Session%206/lr0.5.png)    |   LR=0.5      |
| ![Image 0.2](../../Results/Session%206/lr0.2.png)    |   LR=0.2      | ![Image 0.8](../../Results/Session%206/lr0.8.png)    |   LR=0.8      |
| ![Image 1.0](../../Results/Session%206/lr1.0.png)    |   LR=1.0      | ![Image 2.0](../../Results/Session%206/lr2.0.png)    |   LR=2.0      |

<br>
<br>

# Part 2

# Convolutional Neural Network for MNIST
<br>

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Build Status](https://github.com/TylerYep/torchinfo/actions/workflows/test.yml/badge.svg)](https://github.com/TylerYep/torchinfo/actions/workflows/test.yml)
[![pytorch](https://img.shields.io/badge/pytorch-2.0-orange)](https://pytorch.org/)

## Description

The architecture is a deep convolutional neural network (CNN) which achieved outstanding performance on MNIST image classification. The key characteristic is its simplicity and uniformity in design, making it easy to understand and replicate.

The  architecture consists of several convolutional layers with Batch Normalisation and MaxPooling Operations followed by a GAP Layer.

The core building block is the repeated use of 3x3 convolutional layers (kernels) stacked on top of each other. 


The architecture of the `Net` neural network can be described as follows:

## 1. Convolutional Layers
- `conv1`:
  - 2D Convolutional layer:
    - Input channels: 1
    - Output channels: 16
    - Kernel size: 3
    - Padding: 1
  - ReLU activation function
  - Batch normalization
  - 2D Convolutional layer:
    - Input channels: 16
    - Output channels: 16
    - Kernel size: 3
    - Padding: 1
  - ReLU activation function
  - Batch normalization
  - 2D Max Pooling layer:
    - Kernel size: 2
    - Stride: 2
  - Dropout layer:
    - Dropout rate: 0.25

- `conv2`:
  - 2D Convolutional layer:
    - Input channels: 16
    - Output channels: 16
    - Kernel size: 3
    - Padding: 1
  - ReLU activation function
  - Batch normalization
  - 2D Convolutional layer:
    - Input channels: 16
    - Output channels: 16
    - Kernel size: 3
    - Padding: 1
  - ReLU activation function
  - Batch normalization
  - 2D Max Pooling layer:
    - Kernel size: 2
    - Stride: 2
  - Dropout layer:
    - Dropout rate: 0.25

- `conv3`:
  - 2D Convolutional layer:
    - Input channels: 16
    - Output channels: 16
    - Kernel size: 3
    - Padding: 1
  - ReLU activation function
  - Batch normalization
  - 2D Convolutional layer:
    - Input channels: 16
    - Output channels: 32
    - Kernel size: 3
    - Padding: 1
  - ReLU activation function
  - Batch normalization
  - 2D Max Pooling layer:
    - Kernel size: 2
    - Stride: 2
  - Dropout layer:
    - Dropout rate: 0.25

## 2. Global Average Pooling Layer
- `gap`:
  - 2D Convolutional layer:
    - Input channels: 32
    - Output channels: 10
    - Kernel size: (1, 1)
    - Padding: 0

## 3. Fully Connected Layers
- `fc`:
  - Linear layer:
    - Input size: 90
    - Output size: 10

## 4. Forward Function
- The `forward` method defines the forward pass of the network.
- Input `x` passes through the convolutional layers (`conv1`, `conv2`, `conv3`) and the global average pooling layer (`gap`).
- The output of the last layer is reshaped to match the expected input shape of the linear layer.
- The reshaped output is passed through the fully connected layer (`fc`).
- Finally, the log softmax activation is applied to the output using `F.log_softmax` to obtain predicted probabilities.

<br>

# Model - Architecture

```python
================================================================
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 28, 28]             160
              ReLU-2           [-1, 16, 28, 28]               0
       BatchNorm2d-3           [-1, 16, 28, 28]              32
            Conv2d-4           [-1, 16, 28, 28]           2,320
              ReLU-5           [-1, 16, 28, 28]               0
       BatchNorm2d-6           [-1, 16, 28, 28]              32
         MaxPool2d-7           [-1, 16, 14, 14]               0
           Dropout-8           [-1, 16, 14, 14]               0
            Conv2d-9           [-1, 16, 14, 14]           2,320
             ReLU-10           [-1, 16, 14, 14]               0
      BatchNorm2d-11           [-1, 16, 14, 14]              32
           Conv2d-12           [-1, 16, 14, 14]           2,320
             ReLU-13           [-1, 16, 14, 14]               0
      BatchNorm2d-14           [-1, 16, 14, 14]              32
        MaxPool2d-15             [-1, 16, 7, 7]               0
          Dropout-16             [-1, 16, 7, 7]               0
           Conv2d-17             [-1, 16, 7, 7]           2,320
             ReLU-18             [-1, 16, 7, 7]               0
      BatchNorm2d-19             [-1, 16, 7, 7]              32
           Conv2d-20             [-1, 32, 7, 7]           4,640
             ReLU-21             [-1, 32, 7, 7]               0
      BatchNorm2d-22             [-1, 32, 7, 7]              64
        MaxPool2d-23             [-1, 32, 3, 3]               0
          Dropout-24             [-1, 32, 3, 3]               0
           Conv2d-25             [-1, 10, 3, 3]             330
           Linear-26                   [-1, 10]             910
================================================================
Total params: 15,544
Trainable params: 15,544
Non-trainable params: 0
================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.84
Params size (MB): 0.06
Estimated Total Size (MB): 0.90
================================================================
```

# Key Points

- Less than 20K Parameters
- Higher scores
- Achieved in less than 20 Epochs

# Validation Scores

![epochs](../../Results/Session%206/validationscores.png)


