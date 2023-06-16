# ERA - V1
## Session 7 - In Depth Coding

<br>

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Build Status](https://github.com/TylerYep/torchinfo/actions/workflows/test.yml/badge.svg)](https://github.com/TylerYep/torchinfo/actions/workflows/test.yml)
[![pytorch](https://img.shields.io/badge/pytorch-2.0-orange)](https://pytorch.org/)

<br>

## About the Data 

<br>

- The MNIST dataset consists of a large collection of handwritten digits from ```0 - 9```. 
- It contains *60,000* training examples and *10,000* test examples. 
- Each example is a grayscale image of size ```28x28``` pixels, representing a single handwritten digit. 
- The images are accompanied by corresponding labels indicating the true digit value.

<br>

# Assignment

Your new target is:
- 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
- Less than or equal to 15 Epochs
- Less than 8000 Parameters

<br>

# Solutions

- There are totally 6 Model Notebooks `.ipynb` and `model.py` file which contains the model's architecture and a `utils.py` file which has the functions

<br>

# Model 1

### Target:

- Get the set-up right   
- Set Transforms  
- Set Data Loader  
- Set Basic Working Code   
- Set Basic Training  & Test Loop   

### Results:

- Parameters : 6.3M
- Best Training Accuracy : 99.92
- Best Test Accuracy : 99.30

### Analysis:

- Extremely Heavy Model for such a problem
- Model is over-fitting, but we are changing our model approaches in the next few step

<br>


# Model 2

### Target:

- Try to Get the basic skeleton right. We will try and avoid changing this skeleton as much as possible. 

### Results:

- Parameters : 7,272
- Best Train Accuracy : 97.35
- Best Test Accuracy : 97.93

### Analysis:

- We see some over-fitting
- The model is still around 7.2K parameters , but working fine next step is either to increase capacity or add regularisation and maybe Batch-Normalisation. 

<br>

# Model 3

### Target:

- Add Batch-norm to increase model efficiency.

### Results:

- Parameters : 7,448
- Best Train Accuracy : 99.51
- Best Test Accuracy : 99.30

### Analysis:

- We have started to see over-fitting now. 
- Even if the model is pushed further, it won't be able to get to 99.30

<br>

# Model 4

### Target

- Add Regularization, Dropout

### Results

- Parameters : 7,448
- Best Train Accuracy : 98.78
- Best Test Accuracy : 99.01

### Analysis

- Model can be pushed to learn in more epochs
- Model is slightly underfitting and not overfitting as training and test accuracies are closeby.

<br>

# Model 5

### Target

- Add image augmentation with Random Rotation and Fill to improve the model performance.

### Results

- Parameters : 7,448
- Best Train Accuracy : 98.61
- Best Test Accuracy : 99.07

### Analysis

- Model with 7K parameters is able to reach till 99.07% accuracy in 15 epochs.
- Image augmentation doesn't seem to show much improvement. It may be because of presense of dropout which effectively does similar function.

<br>

# Model 6

### Target   

- Study effect of including StepLR rate scheduler.

### Results  

- Parameters : 7,416
- Best Train Accuracy : 98.93
- Best Test Accuracy : 99.45

### Analysis  

- Finding a good LR schedule is hard.
- Model with 7.4K parameters is cross 99.45% accuracy in 15 epochs.
- Model meets all the requirement of model size, accuracy and epoch.

<br>

# Accuracy 

![Accuracy](../Results/Session%207/acc.png)

<br>

# Receptive Field Calculations:

| Layers   | Kernel | R_in | N_in | J_in | Stride | Padding | R_out | N_out | J_out |
|----------|--------|------|------|------|--------|---------|-------|-------|-------|
| input    |        |      |      |      |        |         | 1     | 28    | 1     |
| conv1    | 3      | 1    | 28   | 1    | 1      | 0       | 3     | 26    | 1     |
| conv2    | 3      | 3    | 26   | 1    | 1      | 0       | 5     | 24    | 1     |
| conv3    | 1      | 5    | 24   | 1    | 1      | 0       | 5     | 24    | 1     |
| maxpool  | 2      | 5    | 24   | 1    | 2      | 0       | 6     | 12    | 2     |
| conv4    | 3      | 6    | 12   | 2    | 1      | 0       | 10    | 10    | 2     |
| conv5    | 3      | 10   | 10   | 2    | 1      | 0       | 14    | 8     | 2     |
| conv6    | 3      | 14   | 8    | 2    | 1      | 0       | 18    | 6     | 2     |
| conv7    | 3      | 18   | 6    | 2    | 1      | 0       | 22    | 4     | 2     |
| avgpool  | 4      | 22   | 4    | 2    | 1      | 0       | 28    | 1     | 2     |
| conv8    | 1      | 28   | 1    | 2    | 1      | 0       | 28    | 1     | 2     |

