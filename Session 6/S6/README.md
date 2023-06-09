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



