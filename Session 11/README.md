# ERA - V1
## Session 11 - CAMs, LRs and Optimizers

<br>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![Albumentations](https://img.shields.io/badge/Albumentations-v1.3.1-yellow)](https://albumentations.ai/docs/)
[![Torchvision 0.15](https://img.shields.io/badge/torchvision-v0.15-green)](https://pytorch.org/vision/stable/index.html)
[![TQDM](https://img.shields.io/badge/tqdm-v4.65.0-yellowgreen)](https://tqdm.github.io/)

<br>

# Assignment (Task)

<br>

* Train ResNet18 on Cifar10 for 20 Epochs.

# Data Augmentation

<br>

```python
def __init__(self):
    self.albumentations_transform = A.Compose([
        A.Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615],always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.PadIfNeeded(40, 40, p=1),
        A.RandomCrop(32, 32, p=1),
        A.PadIfNeeded(64, 64, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16, fill_value=0, p=1),
        A.CenterCrop(32, 32, p=1),
        ToTensorV2()
    ])
```

# Model Summary

<br>

```python
============================================================================================================================================
Layer (type:depth-idx)                   Input Shape               Output Shape              Param #                   Param %
============================================================================================================================================
ResNet                                   [32, 3, 32, 32]           [32, 10]                  --                             --
├─Conv2d: 1-1                            [32, 3, 32, 32]           [32, 64, 32, 32]          1,728                       0.02%
├─BatchNorm2d: 1-2                       [32, 64, 32, 32]          [32, 64, 32, 32]          128                         0.00%
├─Sequential: 1-3                        [32, 64, 32, 32]          [32, 64, 32, 32]          --                             --
│    └─BasicBlock: 2-1                   [32, 64, 32, 32]          [32, 64, 32, 32]          --                             --
│    │    └─Conv2d: 3-1                  [32, 64, 32, 32]          [32, 64, 32, 32]          36,864                      0.33%
│    │    └─BatchNorm2d: 3-2             [32, 64, 32, 32]          [32, 64, 32, 32]          128                         0.00%
│    │    └─Conv2d: 3-3                  [32, 64, 32, 32]          [32, 64, 32, 32]          36,864                      0.33%
│    │    └─BatchNorm2d: 3-4             [32, 64, 32, 32]          [32, 64, 32, 32]          128                         0.00%
│    │    └─Sequential: 3-5              [32, 64, 32, 32]          [32, 64, 32, 32]          --                             --
│    └─BasicBlock: 2-2                   [32, 64, 32, 32]          [32, 64, 32, 32]          --                             --
│    │    └─Conv2d: 3-6                  [32, 64, 32, 32]          [32, 64, 32, 32]          36,864                      0.33%
│    │    └─BatchNorm2d: 3-7             [32, 64, 32, 32]          [32, 64, 32, 32]          128                         0.00%
│    │    └─Conv2d: 3-8                  [32, 64, 32, 32]          [32, 64, 32, 32]          36,864                      0.33%
│    │    └─BatchNorm2d: 3-9             [32, 64, 32, 32]          [32, 64, 32, 32]          128                         0.00%
│    │    └─Sequential: 3-10             [32, 64, 32, 32]          [32, 64, 32, 32]          --                             --
├─Sequential: 1-4                        [32, 64, 32, 32]          [32, 128, 16, 16]         --                             --
│    └─BasicBlock: 2-3                   [32, 64, 32, 32]          [32, 128, 16, 16]         --                             --
│    │    └─Conv2d: 3-11                 [32, 64, 32, 32]          [32, 128, 16, 16]         73,728                      0.66%
│    │    └─BatchNorm2d: 3-12            [32, 128, 16, 16]         [32, 128, 16, 16]         256                         0.00%
│    │    └─Conv2d: 3-13                 [32, 128, 16, 16]         [32, 128, 16, 16]         147,456                     1.32%
│    │    └─BatchNorm2d: 3-14            [32, 128, 16, 16]         [32, 128, 16, 16]         256                         0.00%
│    │    └─Sequential: 3-15             [32, 64, 32, 32]          [32, 128, 16, 16]         --                             --
│    │    │    └─Conv2d: 4-1             [32, 64, 32, 32]          [32, 128, 16, 16]         8,192                       0.07%
│    │    │    └─BatchNorm2d: 4-2        [32, 128, 16, 16]         [32, 128, 16, 16]         256                         0.00%
│    └─BasicBlock: 2-4                   [32, 128, 16, 16]         [32, 128, 16, 16]         --                             --
│    │    └─Conv2d: 3-16                 [32, 128, 16, 16]         [32, 128, 16, 16]         147,456                     1.32%
│    │    └─BatchNorm2d: 3-17            [32, 128, 16, 16]         [32, 128, 16, 16]         256                         0.00%
│    │    └─Conv2d: 3-18                 [32, 128, 16, 16]         [32, 128, 16, 16]         147,456                     1.32%
│    │    └─BatchNorm2d: 3-19            [32, 128, 16, 16]         [32, 128, 16, 16]         256                         0.00%
│    │    └─Sequential: 3-20             [32, 128, 16, 16]         [32, 128, 16, 16]         --                             --
├─Sequential: 1-5                        [32, 128, 16, 16]         [32, 256, 8, 8]           --                             --
│    └─BasicBlock: 2-5                   [32, 128, 16, 16]         [32, 256, 8, 8]           --                             --
│    │    └─Conv2d: 3-21                 [32, 128, 16, 16]         [32, 256, 8, 8]           294,912                     2.64%
│    │    └─BatchNorm2d: 3-22            [32, 256, 8, 8]           [32, 256, 8, 8]           512                         0.00%
│    │    └─Conv2d: 3-23                 [32, 256, 8, 8]           [32, 256, 8, 8]           589,824                     5.28%
│    │    └─BatchNorm2d: 3-24            [32, 256, 8, 8]           [32, 256, 8, 8]           512                         0.00%
│    │    └─Sequential: 3-25             [32, 128, 16, 16]         [32, 256, 8, 8]           --                             --
│    │    │    └─Conv2d: 4-3             [32, 128, 16, 16]         [32, 256, 8, 8]           32,768                      0.29%
│    │    │    └─BatchNorm2d: 4-4        [32, 256, 8, 8]           [32, 256, 8, 8]           512                         0.00%
│    └─BasicBlock: 2-6                   [32, 256, 8, 8]           [32, 256, 8, 8]           --                             --
│    │    └─Conv2d: 3-26                 [32, 256, 8, 8]           [32, 256, 8, 8]           589,824                     5.28%
│    │    └─BatchNorm2d: 3-27            [32, 256, 8, 8]           [32, 256, 8, 8]           512                         0.00%
│    │    └─Conv2d: 3-28                 [32, 256, 8, 8]           [32, 256, 8, 8]           589,824                     5.28%
│    │    └─BatchNorm2d: 3-29            [32, 256, 8, 8]           [32, 256, 8, 8]           512                         0.00%
│    │    └─Sequential: 3-30             [32, 256, 8, 8]           [32, 256, 8, 8]           --                             --
├─Sequential: 1-6                        [32, 256, 8, 8]           [32, 512, 4, 4]           --                             --
│    └─BasicBlock: 2-7                   [32, 256, 8, 8]           [32, 512, 4, 4]           --                             --
│    │    └─Conv2d: 3-31                 [32, 256, 8, 8]           [32, 512, 4, 4]           1,179,648                  10.56%
│    │    └─BatchNorm2d: 3-32            [32, 512, 4, 4]           [32, 512, 4, 4]           1,024                       0.01%
│    │    └─Conv2d: 3-33                 [32, 512, 4, 4]           [32, 512, 4, 4]           2,359,296                  21.11%
│    │    └─BatchNorm2d: 3-34            [32, 512, 4, 4]           [32, 512, 4, 4]           1,024                       0.01%
│    │    └─Sequential: 3-35             [32, 256, 8, 8]           [32, 512, 4, 4]           --                             --
│    │    │    └─Conv2d: 4-5             [32, 256, 8, 8]           [32, 512, 4, 4]           131,072                     1.17%
│    │    │    └─BatchNorm2d: 4-6        [32, 512, 4, 4]           [32, 512, 4, 4]           1,024                       0.01%
│    └─BasicBlock: 2-8                   [32, 512, 4, 4]           [32, 512, 4, 4]           --                             --
│    │    └─Conv2d: 3-36                 [32, 512, 4, 4]           [32, 512, 4, 4]           2,359,296                  21.11%
│    │    └─BatchNorm2d: 3-37            [32, 512, 4, 4]           [32, 512, 4, 4]           1,024                       0.01%
│    │    └─Conv2d: 3-38                 [32, 512, 4, 4]           [32, 512, 4, 4]           2,359,296                  21.11%
│    │    └─BatchNorm2d: 3-39            [32, 512, 4, 4]           [32, 512, 4, 4]           1,024                       0.01%
│    │    └─Sequential: 3-40             [32, 512, 4, 4]           [32, 512, 4, 4]           --                             --
├─Linear: 1-7                            [32, 512]                 [32, 10]                  5,130                       0.05%
============================================================================================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
Total mult-adds (G): 17.77
============================================================================================================================================
Input size (MB): 0.39
Forward/backward pass size (MB): 314.58
Params size (MB): 44.70
Estimated Total Size (MB): 359.66
============================================================================================================================================
```

<br>

# Train Data

<br>

![train_data](../Results/Session%2011/train.png)

<br>

# Test Data

<br>

![test_data](../Results/Session%2011/test.png)

<br>

# One Cycle LR Policy

<br>

* Sets the learning rate of each parameter group according to the 1cycle learning rate policy. 
* The 1cycle policy anneals the learning rate from an initial learning rate to some maximum learning rate and then from that maximum learning rate to some minimum learning rate much lower than the initial learning rate. 

<br>

![one_cycle](../Results/Session%2011/lr.png)

<br>

# Results

<br>

![train](../Results/Session%2011/train_.png)
![test](../Results/Session%2011/test_.png)

<br>

# Mis-Classified Images

<br>

![mis](../Results/Session%2011/mis.png)

<br>

# GradCam 

<br>

![gradCam](../Results/Session%2011/gradcam.png)

<br>

# All Class Accuracy

<br>

```python
Accuracy for class: plane is 94.1 %
Accuracy for class: car   is 97.2 %
Accuracy for class: bird  is 88.2 %
Accuracy for class: cat   is 82.8 %
Accuracy for class: deer  is 94.1 %
Accuracy for class: dog   is 87.4 %
Accuracy for class: frog  is 94.7 %
Accuracy for class: horse is 95.5 %
Accuracy for class: ship  is 96.3 %
Accuracy for class: truck is 95.8 %
```

<br>

# [Training Logs](training_logs.md)