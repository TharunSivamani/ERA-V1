# Session 12

# PyTorch Lightning and AI Application Development

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch-Lightning](https://img.shields.io/badge/pytorch_lightning-v2.0.6-red)](https://lightning.ai/docs/pytorch/latest/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![Torchvision 0.15](https://img.shields.io/badge/torchvision-v0.15-green)](https://pytorch.org/vision/stable/index.html)
[![TQDM](https://img.shields.io/badge/tqdm-v4.65.0-yellowgreen)](https://tqdm.github.io/)
[![Albumentations](https://img.shields.io/badge/Albumentations-v1.3.1-yellow)](https://albumentations.ai/docs/)
[![Torch-LR-Finder](https://img.shields.io/badge/TorchLRFinder-v0.2.1-red)](https://pypi.org/project/torch-lr-finder/)

<br>

# Task

1. Move your S10 assignment to Lightning first and then to Spaces such that:
- (You have retrained your model on Lightning)
- You are using Gradio
- Your spaces app has these features:
    - ask the user whether he/she wants to see GradCAM images and how many, and from which layer, allow opacity change as well
    - ask whether he/she wants to view misclassified images, and how many
    - allow users to upload new images, as well as provide 10 example images
    - ask how many top classes are to be shown (make sure the user cannot enter more than 10)
- Add the full details on what your App is doing to Spaces README 

# Solution

This repository contains a custom `Resnet18` Pytorch model trained and validated on `CIFAR-10` dataset. The scheduler used here is `OneCycleLR`.

<br>

# Applying Albumentations library

<br>

```python
class a_train_transform:
    def __init__(self):
        self.albumentations_transform = A.Compose(
            [
                A.Normalize(
                    mean=[0.4914, 0.4822, 0.4471], std=[0.2469, 0.2433, 0.2615]
                ),
                A.HorizontalFlip(p=0.5),
                A.PadIfNeeded(40, 40, p=1),
                A.RandomCrop(32, 32, p=1),
                A.PadIfNeeded(64, 64, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
                A.CoarseDropout(
                    max_holes=1, max_height=16, max_width=16, fill_value=0, p=1
                ),
                A.CenterCrop(32, 32, p=1),
                ToTensorV2(),
            ]
        )

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)["image"]
        return img

```

<br>

# Train Data 

<br>

![train_img](../Results/Session%2012/train.png)

<br>

# Model Summary

<br>

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,792
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
           Dropout-4           [-1, 64, 32, 32]               0
         ConvBlock-5           [-1, 64, 32, 32]               0
            Conv2d-6          [-1, 128, 32, 32]          73,856
         MaxPool2d-7          [-1, 128, 16, 16]               0
       BatchNorm2d-8          [-1, 128, 16, 16]             256
              ReLU-9          [-1, 128, 16, 16]               0
          Dropout-10          [-1, 128, 16, 16]               0
  DownsampleLayer-11          [-1, 128, 16, 16]               0
           Conv2d-12          [-1, 128, 16, 16]         147,584
      BatchNorm2d-13          [-1, 128, 16, 16]             256
             ReLU-14          [-1, 128, 16, 16]               0
          Dropout-15          [-1, 128, 16, 16]               0
        ConvBlock-16          [-1, 128, 16, 16]               0
           Conv2d-17          [-1, 128, 16, 16]         147,584
      BatchNorm2d-18          [-1, 128, 16, 16]             256
             ReLU-19          [-1, 128, 16, 16]               0
          Dropout-20          [-1, 128, 16, 16]               0
        ConvBlock-21          [-1, 128, 16, 16]               0
    ResidualBlock-22          [-1, 128, 16, 16]               0
        MakeLayer-23          [-1, 128, 16, 16]               0
           Conv2d-24          [-1, 256, 16, 16]         295,168
        MaxPool2d-25            [-1, 256, 8, 8]               0
      BatchNorm2d-26            [-1, 256, 8, 8]             512
             ReLU-27            [-1, 256, 8, 8]               0
          Dropout-28            [-1, 256, 8, 8]               0
  DownsampleLayer-29            [-1, 256, 8, 8]               0
        MakeLayer-30            [-1, 256, 8, 8]               0
           Conv2d-31            [-1, 512, 8, 8]       1,180,160
        MaxPool2d-32            [-1, 512, 4, 4]               0
      BatchNorm2d-33            [-1, 512, 4, 4]           1,024
             ReLU-34            [-1, 512, 4, 4]               0
          Dropout-35            [-1, 512, 4, 4]               0
  DownsampleLayer-36            [-1, 512, 4, 4]               0
           Conv2d-37            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-38            [-1, 512, 4, 4]           1,024
             ReLU-39            [-1, 512, 4, 4]               0
          Dropout-40            [-1, 512, 4, 4]               0
        ConvBlock-41            [-1, 512, 4, 4]               0
           Conv2d-42            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-43            [-1, 512, 4, 4]           1,024
             ReLU-44            [-1, 512, 4, 4]               0
          Dropout-45            [-1, 512, 4, 4]               0
        ConvBlock-46            [-1, 512, 4, 4]               0
    ResidualBlock-47            [-1, 512, 4, 4]               0
        MakeLayer-48            [-1, 512, 4, 4]               0
        MaxPool2d-49            [-1, 512, 1, 1]               0
          Flatten-50                  [-1, 512]               0
           Linear-51                   [-1, 10]           5,130
================================================================
Total params: 6,575,370
Trainable params: 6,575,370
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 10.32
Params size (MB): 25.08
Estimated Total Size (MB): 35.42
----------------------------------------------------------------
```

<br>

# Finding Optimal LR

<br>

![lr](../Results/Session%2012/lr.png)

<br>

# Misclassified Images

<br>

![mis](../Results/Session%2012/mis.png)

<br>

# Results

<br>

![plots](../Results/Session%2012/plots.png)

<br>

# GRADCAM Images - Layer 3

<br>

![gradcam](../Results/Session%2012/gradcam.png)

<br>

# Training - Validation [Logs](training_logs.md)