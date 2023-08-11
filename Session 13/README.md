# ERA - V1

## Session 13 - YOLO and Object Detection Techniques - Part 1 & 2

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch-Lightning](https://img.shields.io/badge/pytorch_lightning-v2.0.6-red)](https://lightning.ai/docs/pytorch/latest/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![Torchvision 0.15](https://img.shields.io/badge/torchvision-v0.15-green)](https://pytorch.org/vision/stable/index.html)
[![TQDM](https://img.shields.io/badge/tqdm-v4.65.0-yellowgreen)](https://tqdm.github.io/)
[![Albumentations](https://img.shields.io/badge/Albumentations-v1.3.1-yellow)](https://albumentations.ai/docs/)
[![Torch-LR-Finder](https://img.shields.io/badge/TorchLRFinder-v0.2.1-red)](https://pypi.org/project/torch-lr-finder/)

<br>

# Task

1. Move the code to PytorchLightning
2. Train the model to reach such that all of these are true:
* Class accuracy is more than 75%
* No Obj accuracy of more than 95%
* Object Accuracy of more than 70% (assuming you had to reduce the kernel numbers, else 80/98/78)
* Ideally trailed till 40 epochs
3. Add these training features:
* Add multi-resolution training - the code shared trains only on one resolution 416
* Add Implement Mosaic Augmentation only 75% of the times
* Train on float16
* GradCam must be implemented.
4. Things that are allowed due to HW constraints:
* Change of batch size
* Change of resolution
* Change of OCP parameters

<br>

# Solution

This repository contains a `DARKNET` Pytorch Architecture Model trained and validated on `PASCAL_VOC` dataset having `20` classes. The scheduler used here is `OneCycleLR`.

<br>

# Applying Albumentations library

```python
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.Rotate(limit = 10, interpolation=1, border_mode=4),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
                # A.Affine(shear=15, p=0.5, mode="constant"),
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)
```

```python
test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)
```

<br>