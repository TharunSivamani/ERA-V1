# ERA - V1
## Session 8 - Batch Normalization & Regularization

<br>

# Task

* Dataset: CIFAR-10
* Network Architecture: C1, C2, C3, P1, C3, C4, C5, C6, P2, C7, C8, C9, GAP, C10
* Maximum Parameter Count: 50,000
* Layer Addition: One layer can be added to another
* Max Epochs: 20

<br>

# Solution

This repository contains the various types of normalization model's trained on `CIFAR10` dataset 

1. `Batch Normalization`
2. `Group Normalization`
3. `Layer Normalization`

<br>

## Code Structure

1. `model.py` - contains the models used in (Session 8 , Session 7 , Session 6).

2. `utils.py` - contains all the python scripts needed for basic functions

3. `backprop.py` - contains the train and test class modules

4. `dataloader.py` - contains the initial dataset loading and train and test loaders initialization

<br>

# Group Normalisation 

![GN](../Results/Session%208/gn.png)

Parameters : 47,818
Best Train Accuracy : 76.55
Best Test Accuracy : 72.54

## Misclassified Images

![Gn](../Results/Session%208/gn_miss.png)

# Batch Normalisation 

![BN](../Results/Session%208/bn.png)

Parameters : 47,818
Best Train Accuracy : 81.27
Best Test Accuracy : 76.70

## Misclassified Images

![Bn](../Results/Session%208/bn_miss.png)

# Layer Normalisation

<br>

![LN](../Results/Session%208/ln.png)

Parameters : 47,818
Best Train Accuracy : 76.24
Best Test Accuracy : 72.98

<br>

## Misclassified Images

![Ln](../Results/Session%208/ln_miss.png)

<br>

# RF Calculations

<br>

| Layers   | kernel | R_in | N_in | J_in | Stride | Padding | R_out | N_out | J_out |
|----------|--------|------|------|------|--------|---------|-------|-------|-------|
| input    |        |      |      |      |        |         | 1     | 32    | 1     |
| Conv1    | 3      | 1    | 32   | 1    | 1      | 0       | 3     | 30    | 1     |
| Conv2    | 3      | 3    | 30   | 1    | 1      | 0       | 5     | 28    | 1     |
| conv3    | 1      | 5    | 28   | 1    | 1      | 0       | 5     | 28    | 1     |
| maxpool  | 2      | 5    | 28   | 1    | 2      | 0       | 6     | 14    | 2     |
| Conv3a   | 3      | 6    | 14   | 2    | 1      | 1       | 10    | 14    | 2     |
| Conv4    | 3      | 10   | 14   | 2    | 1      | 1       | 14    | 14    | 2     |
| Conv5    | 3      | 14   | 14   | 2    | 1      | 1       | 18    | 14    | 2     |
| conv6    | 1      | 18   | 14   | 2    | 1      | 0       | 18    | 14    | 2     |
| maxpool  | 2      | 18   | 14   | 2    | 2      | 0       | 20    | 7     | 4     |
| Conv7    | 3      | 20   | 7    | 4    | 1      | 1       | 28    | 7     | 4     |
| Conv8    | 3      | 28   | 7    | 4    | 1      | 1       | 36    | 7     | 4     |
| Conv9    | 3      | 36   | 7    | 4    | 1      | 1       | 44    | 7     | 4     |
| GAP      | 7      | 44   | 7    | 4    | 7      | 0       | 68    | 1     | 28    |
| conv10   | 1      | 68   | 1    | 28   | 1      | 0       | 68    | 1     | 28    |


<br>


# Regularization

## L1 Regularization (Lasso Regression)

- L1 regularization adds an L1 penalty equal to the absolute value of the magnitude of coefficients. 
- When our input features have weights closer to zero this leads to a sparse L1 norm. 
- In the Sparse solution, the majority of the input features have zero weights and
very few features have non-zero weights.

<br>

![L1](../Results/Session%208/L1.jpg)

<br>

## L2 Regularization (Ridge Regularization)

- L2 regularization is similar to L1 regularization. 
- But it adds a squared magnitude of coefficient as a penalty term to the loss function. 
- L2 will not yield sparse models and all coefficients are shrunk by the same factor.

<br>

![L2](../Results/Session%208/L2.jpg)