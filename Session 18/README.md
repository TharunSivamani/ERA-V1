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