# ERA - V1
## Session 10 - Residual Connections in CNNs and One Cycle Policy!

<br>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![Albumentations](https://img.shields.io/badge/Albumentations-v1.3.1-yellow)](https://albumentations.ai/docs/)
[![Torchvision 0.15](https://img.shields.io/badge/torchvision-v0.15-green)](https://pytorch.org/vision/stable/index.html)
[![TQDM](https://img.shields.io/badge/tqdm-v4.65.0-yellowgreen)](https://tqdm.github.io/)

<br>

# Assignment (Task)

<br>

* `ResNet` architecture for `CIFAR10` that has the following architecture:
1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
2. Layer1 -
* X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
* R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
Add(X, R1)
3. Layer 2 -
* Conv 3x3 [256k]
* MaxPooling2D
* BN
* ReLU
4. Layer 3 -
* X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
* R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
Add(X, R2)
5. MaxPooling with Kernel Size 4
6. FC Layer 
7. SoftMax
8. Uses One Cycle Policy such that:
* Total Epochs = 24
* Max at Epoch = 5
* LRMIN = FIND
* LRMAX = FIND
* NO Annihilation
9. Uses this transform -
```python
RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
```
10. Batch size = `512`
11. Use ADAM, and CrossEntropyLoss
12. Target Accuracy: `90%`

<br>

# Data Augmentation

<br>

```python
def __init__(self):
   self.albumentations_transform = A.Compose([
      A.Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615], always_apply=True),
      A.PadIfNeeded(min_height=36, min_width=36, always_apply=True),
      A.RandomCrop(height=32, width=32, always_apply=True),
      A.HorizontalFlip(),
      A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=0, p=1.0),
      ToTensorV2(),
   ])
```

<br>

# Data

<br>

![train_data](../Results/Session%2010/train.png)

<br>

![test_data](../Results/Session%2010/test.png)

<br>

# Model Summary

<br>

```python
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
           Dropout-4           [-1, 64, 32, 32]               0
         ConvLayer-5           [-1, 64, 32, 32]               0
 Make_Custom_Layer-6           [-1, 64, 32, 32]               0
            Conv2d-7          [-1, 128, 32, 32]          73,728
         MaxPool2d-8          [-1, 128, 16, 16]               0
       BatchNorm2d-9          [-1, 128, 16, 16]             256
             ReLU-10          [-1, 128, 16, 16]               0
          Dropout-11          [-1, 128, 16, 16]               0
        ConvLayer-12          [-1, 128, 16, 16]               0
           Conv2d-13          [-1, 128, 16, 16]         147,456
      BatchNorm2d-14          [-1, 128, 16, 16]             256
             ReLU-15          [-1, 128, 16, 16]               0
          Dropout-16          [-1, 128, 16, 16]               0
        ConvLayer-17          [-1, 128, 16, 16]               0
           Conv2d-18          [-1, 128, 16, 16]         147,456
      BatchNorm2d-19          [-1, 128, 16, 16]             256
             ReLU-20          [-1, 128, 16, 16]               0
          Dropout-21          [-1, 128, 16, 16]               0
        ConvLayer-22          [-1, 128, 16, 16]               0
Make_Custom_Layer-23          [-1, 128, 16, 16]               0
           Conv2d-24          [-1, 256, 16, 16]         294,912
        MaxPool2d-25            [-1, 256, 8, 8]               0
      BatchNorm2d-26            [-1, 256, 8, 8]             512
             ReLU-27            [-1, 256, 8, 8]               0
          Dropout-28            [-1, 256, 8, 8]               0
        ConvLayer-29            [-1, 256, 8, 8]               0
Make_Custom_Layer-30            [-1, 256, 8, 8]               0
           Conv2d-31            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-32            [-1, 512, 4, 4]               0
      BatchNorm2d-33            [-1, 512, 4, 4]           1,024
             ReLU-34            [-1, 512, 4, 4]               0
          Dropout-35            [-1, 512, 4, 4]               0
        ConvLayer-36            [-1, 512, 4, 4]               0
           Conv2d-37            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-38            [-1, 512, 4, 4]           1,024
             ReLU-39            [-1, 512, 4, 4]               0
          Dropout-40            [-1, 512, 4, 4]               0
        ConvLayer-41            [-1, 512, 4, 4]               0
           Conv2d-42            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-43            [-1, 512, 4, 4]           1,024
             ReLU-44            [-1, 512, 4, 4]               0
          Dropout-45            [-1, 512, 4, 4]               0
        ConvLayer-46            [-1, 512, 4, 4]               0
Make_Custom_Layer-47            [-1, 512, 4, 4]               0
        MaxPool2d-48            [-1, 512, 1, 1]               0
          Flatten-49                  [-1, 512]               0
           Linear-50                   [-1, 10]           5,130
================================================================
Total params: 6,573,130
Trainable params: 6,573,130
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 10.51
Params size (MB): 25.07
Estimated Total Size (MB): 35.59
----------------------------------------------------------------
```

<br>

# One Cycle LR Policy

<br>

* Sets the learning rate of each parameter group according to the 1cycle learning rate policy. 
* The 1cycle policy anneals the learning rate from an initial learning rate to some maximum learning rate and then from that maximum learning rate to some minimum learning rate much lower than the initial learning rate. 

<br>

![one_cycle](../Results/Session%2010/one_cycle.png)

<br>

# Results

<br>

![results](../Results/Session%2010/model_viz.png)

<br>

# Mis-Classified Images

<br>

![mis](../Results/Session%2010/mis.png)

<br>

# All Class Accuracy

<br>

```python
Accuracy for class: plane is 94.3 %
Accuracy for class: car   is 97.8 %
Accuracy for class: bird  is 91.5 %
Accuracy for class: cat   is 84.1 %
Accuracy for class: deer  is 94.1 %
Accuracy for class: dog   is 88.7 %
Accuracy for class: frog  is 95.5 %
Accuracy for class: horse is 94.6 %
Accuracy for class: ship  is 95.7 %
Accuracy for class: truck is 94.7 %
```

<br>

# [Training Logs](training_logs.md)