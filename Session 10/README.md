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