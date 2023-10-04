# ERA V1

## Session 19   

# CLIP Models and Training them

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![PyTorch-Lightning](https://img.shields.io/badge/pytorch_lightning-v2.0.6-orange)](https://lightning.ai/docs/pytorch/latest/)
[![Torchvision 0.15](https://img.shields.io/badge/torchvision-v0.15-green)](https://pytorch.org/vision/stable/index.html)
[![Albumentations](https://img.shields.io/badge/Albumentations-v1.3.1-yellow)](https://albumentations.ai/docs/)
[![Transformers](https://img.shields.io/badge/Transformers-v4.34.0-lightgreen)](https://huggingface.co/docs/transformers/index)

<br>

# Task

<br>

- Make a CLIP or FastSAM application on gradio/spaces using open-source models.

<br>

# Solution

<br>

This repository contains a Jupyter Notebook which trained a CLIP model for 2 epochs.

<br>

# CLIP

<br>

- The core idea of the CLIP paper is essentially to learn visual representation from the massive corpus of natural language data. 
- The paper showed that a simple pre-training task is sufficient to achieve a competitive performance boost in zero-shot learning.
 
<br>

The **objective of the CLIP model** can be understood as follows:    

```Given an image, a set of 32,768 sampled text snippets was paired with it in our dataset. For example, given a task to predict a number from an image, the model is likely to predict that “the number is one” or, “the number is two”, or “the number is XYZ” and so on.```

<br>

# Segment Anything

<br>

SAM's design hinges on three main components:

- the **promptable segmentation task** to enable zero-shot generalization
- the **model architecture**.
- the **dataset** that powers the task and model.

<br>

## Model Architecture

<br>

SAM's architecture comprises three components that work together to return a valid segmentation mask:

- An **image encoder** to generate one-time image embeddings.
- A **prompt encoder** that embeds the prompts.
- A **lightweight mask decoder** that combines the embeddings from the prompt and image encoders.
