# ERA - V1
## Session 9 - Advanced Convolutions, Data Augmentation and Visualization

<br>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![Albumentations](https://img.shields.io/badge/Albumentations-v1.3.1-yellow)](https://albumentations.ai/docs/)
[![Torchvision 0.15](https://img.shields.io/badge/torchvision-v0.15-green)](https://pytorch.org/vision/stable/index.html)
[![TQDM](https://img.shields.io/badge/tqdm-v4.65.0-yellowgreen)](https://tqdm.github.io/)

<br>

`Used Dilated Kernels and DepthWise Separable Convolution's for the Solution`

<br>

# Task

Write a new network that 
1. Has the architecture to `C1C2C3C40` (No MaxPooling, but 3 convolutions, where the last one has a stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
2. total RF must be more than `44`
3. one of the layers must use `Depthwise Separable Convolution`
4. one of the layers must use `Dilated Convolution`
5. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
6. use `albumentation` library and apply:
    - horizontal flip
    - shiftScaleRotate
    - coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
7. achieve `85`% accuracy, as many epochs as you want. Total Params to be less than `200k`.

<br>

# Task

Write a new network that 
1. Has the architecture to `C1C2C3C40` (No MaxPooling, but 3 convolutions, where the last one has a stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
2. total RF must be more than `44`
3. one of the layers must use `Depthwise Separable Convolution`
4. one of the layers must use `Dilated Convolution`
5. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
6. use `albumentation` library and apply:
    - horizontal flip
    - shiftScaleRotate
    - coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
7. achieve `85`% accuracy, as many epochs as you want. Total Params to be less than `200k`.

<br>

# Solution

- Used the `CIFAR10` Dataset for the whole experiment

<br>

# File Contents

1. `model.py` - This file contains a model created using Dilated Kernels and Depthwise Separable Convolution , applying skip connections in forward function.

2. `utils.py` - This file contains all the basic functions like getting device , denormalising images , visualizing data functions

3. `backprop.py` - This file contains necessary train and test functions for the model.

4. `dataset.py` - This file contains data loaders and data augmentation methods applied to the data and data loaders.

<br>

![img](../Results/Session%209/img.png)

<br>

# Albumentation - [Docs](https://albumentations.ai/docs/)
<br>

```python
import albumentations as A

def __init__(self):
        self.albumentations_transform = A.Compose([
            A.HorizontalFlip(),
            A.ShiftScaleRotate(),
            A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1,
                            min_height=16, min_width=1, fill_value=(0.49139968, 0.48215827, 0.44653124),mask_fill_value = None),
            A.Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615]),
            ToTensorV2()
        ])
```

<br>

![viz](../Results/Session%209/viz.png)

<br>

# Forward function - (Skip connection)

The Skip connection is implemented in the model's forward function as below.

```python
def forward(self, x):

    x = self.conv1(x)
    x = self.trans1(x)
    x = x + self.conv2(x)
    x = self.trans2(x)
    x = x + self.conv3(x)
    x = self.trans3(x)
    x = x + self.conv4(x)
    x = self.trans4(x)
    x = self.output(x)

    return x
```

<br>

# Model Summary

```python
================================================================
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 24, 32, 32]             648
       BatchNorm2d-2           [-1, 24, 32, 32]              48
           Dropout-3           [-1, 24, 32, 32]               0
              ReLU-4           [-1, 24, 32, 32]               0
            Conv2d-5           [-1, 24, 32, 32]             216
       BatchNorm2d-6           [-1, 24, 32, 32]              48
           Dropout-7           [-1, 24, 32, 32]               0
              ReLU-8           [-1, 24, 32, 32]               0
            Conv2d-9           [-1, 24, 32, 32]             576
           Conv2d-10           [-1, 32, 30, 30]           6,912
      BatchNorm2d-11           [-1, 32, 30, 30]              64
          Dropout-12           [-1, 32, 30, 30]               0
             ReLU-13           [-1, 32, 30, 30]               0
           Conv2d-14           [-1, 32, 30, 30]             288
      BatchNorm2d-15           [-1, 32, 30, 30]              64
          Dropout-16           [-1, 32, 30, 30]               0
             ReLU-17           [-1, 32, 30, 30]               0
           Conv2d-18           [-1, 32, 30, 30]           1,024
           Conv2d-19           [-1, 32, 30, 30]             288
      BatchNorm2d-20           [-1, 32, 30, 30]              64
          Dropout-21           [-1, 32, 30, 30]               0
             ReLU-22           [-1, 32, 30, 30]               0
           Conv2d-23           [-1, 32, 30, 30]           1,024
           Conv2d-24           [-1, 64, 26, 26]          18,432
      BatchNorm2d-25           [-1, 64, 26, 26]             128
          Dropout-26           [-1, 64, 26, 26]               0
             ReLU-27           [-1, 64, 26, 26]               0
           Conv2d-28           [-1, 64, 26, 26]             576
      BatchNorm2d-29           [-1, 64, 26, 26]             128
          Dropout-30           [-1, 64, 26, 26]               0
             ReLU-31           [-1, 64, 26, 26]               0
           Conv2d-32           [-1, 64, 26, 26]           4,096
           Conv2d-33           [-1, 64, 26, 26]             576
      BatchNorm2d-34           [-1, 64, 26, 26]             128
          Dropout-35           [-1, 64, 26, 26]               0
             ReLU-36           [-1, 64, 26, 26]               0
           Conv2d-37           [-1, 64, 26, 26]           4,096
           Conv2d-38           [-1, 96, 18, 18]          55,296
      BatchNorm2d-39           [-1, 96, 18, 18]             192
          Dropout-40           [-1, 96, 18, 18]               0
             ReLU-41           [-1, 96, 18, 18]               0
           Conv2d-42           [-1, 96, 18, 18]             864
      BatchNorm2d-43           [-1, 96, 18, 18]             192
          Dropout-44           [-1, 96, 18, 18]               0
             ReLU-45           [-1, 96, 18, 18]               0
           Conv2d-46           [-1, 96, 18, 18]           9,216
           Conv2d-47           [-1, 96, 18, 18]             864
      BatchNorm2d-48           [-1, 96, 18, 18]             192
          Dropout-49           [-1, 96, 18, 18]               0
             ReLU-50           [-1, 96, 18, 18]               0
           Conv2d-51           [-1, 96, 18, 18]           9,216
           Conv2d-52             [-1, 96, 2, 2]          82,944
      BatchNorm2d-53             [-1, 96, 2, 2]             192
          Dropout-54             [-1, 96, 2, 2]               0
             ReLU-55             [-1, 96, 2, 2]               0
AdaptiveAvgPool2d-56             [-1, 96, 1, 1]               0
           Conv2d-57             [-1, 10, 1, 1]             970
          Flatten-58                   [-1, 10]               0
       LogSoftmax-59                   [-1, 10]               0
================================================================
Total params: 199,562
Trainable params: 199,562
Non-trainable params: 0
================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 12.72
Params size (MB): 0.76
Estimated Total Size (MB): 13.49
================================================================
```
<br>

# Results

<br>

Best Training Accuracy : `81.04`  

Best Testing Accuracy  : `85.08`   

<br>

![res](../Results/Session%209/graphs.png)

<br>

# Misclassified Images
<br>

![mis](../Results/Session%209/misclassified.png)

<br>

# Receptive Field Calculations

<br>

| Block  | Conv layer | Layers | Kernel | R_in | N_in | J_in | Stride | Padding | Dilation | Eff. Kernel size | R_out | N_out | J_out |
|--------|------------|--------|--------|------|------|------|--------|---------|----------|------------------|-------|-------|-------|
| Input  |       |       |       |     |     |     |       |        |         |                 | 1     | 32    | 1     |
| C1 | conv1      | 1      | 3      | 1    | 32   | 1    | 1      | 1       | 1        | 3                | 3     | 32    | 1     |
|        | conv2      | 2      | 3      | 3    | 32   | 1    | 1      | 1       | 1        | 3                | 5     | 32    | 1     |
|        | trans1     | 3      | 3      | 5    | 32   | 1    | 1      | 0       | 1        | 3                | 7     | 30    | 1     |
| C2 | conv3      | 4      | 3      | 7    | 30   | 1    | 1      | 1       | 1        | 3                | 9     | 30    | 1     |
|        | conv4      | 5      | 3      | 9    | 30   | 1    | 1      | 1       | 1        | 3                | 11    | 30    | 1     |
|        | trans2     | 6      | 3      | 11   | 30   | 1    | 1      | 0       | 2        | 5                | 15    | 26    | 1     |
| C3 | conv5      | 7      | 3      | 15   | 26   | 1    | 1      | 1       | 1        | 3                | 17    | 26    | 1     |
|        | conv6      | 8      | 3      | 17   | 26   | 1    | 1      | 1       | 1        | 3                | 19    | 26    | 1     |
|        | trans3     | 9      | 3      | 19   | 26   | 1    | 1      | 0       | 4        | 9                | 27    | 18    | 1     |
| C4 | conv7      | 10     | 3      | 27   | 18   | 1    | 1      | 1       | 1        | 3                | 29    | 18    | 1     |
|        | conv8      | 11     | 3      | 29   | 18   | 1    | 1      | 1       | 1        | 3                | 31    | 18    | 1     |
|        | trans4     | 12     | 3      | 31   | 18   | 1    | 1      | 0       | 8        | 17               | 47    | 2     | 1     |
| GAP    | gap        | 13     | 2      | 47   | 2    | 1    | 1      | 0       | 1        | 2                | 48    | 1     | 1     |
| O | output     | 14     | 1      | 48   | 1    | 1    | 1      | 0       | 1        | 1                | 48    | 1     | 1     |

<br>

# Training Logs - [Link](./training_logs.md)