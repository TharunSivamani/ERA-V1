# ERA - V1

## Session 18 - UNETs, Variational AutoEncoders, and Applications

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![Torchvision 0.15](https://img.shields.io/badge/torchvision-v0.15-green)](https://pytorch.org/vision/stable/index.html)
[![PyTorch-Lightning](https://img.shields.io/badge/pytorch_lightning-v2.0.6-red)](https://lightning.ai/docs/pytorch/latest/)

<br>

# Task

## First Part

<br>

First part of your assignment is to train your own UNet from scratch, you can use the `OxfordIIIT Pet Dataset` and train it 4 times:

1. MP + Tr + BCE
2. MP + Tr + Dice Loss
3. StrConv + Tr + BCE
4. StrConv + Ups + Dice Loss

<br>

## Second Part

<br>

Design a variation of a VAE that takes in two inputs: (image, label)

1. MNIST
* an MNIST image, and
* its label (one hot encoded vector sent through an embedding layer)

Training as you would train a VAE  
Now randomly send an MNIST image, but with a wrong label. 
Do this 25 times, and share what the VAE makes (25 images stacked in 1 image)!     
Now do this for CIFAR10 and share 25 images (1 stacked image)!

2. CIFAR10
* an CIFAR10 image, and
* its label (one hot encoded vector sent through an embedding layer)

<br>

# Solution

<br>

# UNET 

<br>

![unet](../Results/Session%2018/unet.gif)

<br>

* U-Net is a convolutional neural network that was developed for biomedical image segmentation at the Computer Science Department of the University of Freiburg. 
* The network is based on the fully Convolutional neural network and its architecture was modified and extended to work with fewer training images and to yield more precise segmentation.

<br>

# Segmentation Tasks

<br>

## Binary Semantic Segmentation

<br>

![cat](../Results/Session%2018/cat_mask.jpg)

<br>

## Multilevel Semantic Segmentation

<br>

![cycle](../Results/Session%2018/cycle.png)

<br>

# Dice Loss

<br>

```python
def dice_loss(pred, target):
    smooth = 1e-5
    
    # flatten predictions and targets
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return 1 - dice 
```

<br>

# Auto Encoders

<br>

Autoencoders are a type of neural network architecture used for unsupervised learning, where the goal is to learn a compressed representation (encoding) of the input data. The basic idea behind autoencoders is to learn a function that maps the input data to a lower-dimensional representation and then reconstructs the original data from the encoded representation. 

<br>

![ae](../Results/Session%2018/ae.png)

<br>

# Variational Auto Encoders

<br>

Variational Autoencoders (VAEs) have one fundamentally unique property that separates them from vanilla autoencoders, and it is this property that makes them so useful for generative modeling: their latent spaces are, by design, continuous, allowing easy random sampling and interpolation. 

<br>

![vae](../Results/Session%2018/vae.png)

<br>

# Latent Space

<br>

![vael](../Results/Session%2018/vael.png)

<br>

# Loss

<br>

```python
def gaussian_likelihood(self, mean, logscale, sample):
    scale = torch.exp(logscale)
    dist = torch.distributions.Normal(mean, scale)
    log_pxz = dist.log_prob(sample)
    return log_pxz.sum(dim=(1, 2, 3))
```

<br>

```python
def kl_divergence(self, z, mu, std):
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)
    kl = kl.sum(-1)
    return kl
```

# Results

<br>

# MNIST

<br>

![mnist](../../Results/Session%2018/mnist.png)

<br>

# CIFAR10

<br>

![cifar](../../Results/Session%2018/cifar.png)

<br>