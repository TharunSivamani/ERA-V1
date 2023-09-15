# ERA - V1

## Session 17 - BERT, GPT & ViT

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)

# Task

1. Re-write the code in such a way where there is **one transformer.py file** that you can use to train all 3 models.

<br>

# Solution

<br>

* This repository contains `3 ipynb files` used for training `BERT, GPT and ViT models`.

<br>

# Transformer

<br>

![Transformer-Encoder-Decoder](../Results/Session%2017/transformer-encoder-decoder.png)

<br>

# BERT

<br>

- **Bidirectional Encoder Representations** from Transformers, is a State-Of-The-Art Natural Language Processing (NLP) model that was introduced by **Google AI researchers** in **2018**.   
- BERT is basically just the encoder part of the Transformer.
- In BERT we used the MASK token to predict the missing work.    

<br>

![BERT](../Results/Session%2017/bert.png)

<br>

# GPT

<br>

- **Generative Pre-trained Transformer** is a family of State-Of-The-Art Natural Language Processing (NLP) models developed by **OpenAI**.    
- GPT is just the decoder part of the network. GPT will output one token at a time.   
- In GPT, we will MASK "all" future words. So our attention would be "masked" attention.   
- GPT models are based on the Transformer architecture and are known for their ability to generate coherent and contextually relevant text.   

<br>

![GPT](../Results/Session%2017/gpt.png)

<br>

# ViT

<br>

- The **Vision Transformer** is a model for image classification that employs a Transformer-like architecture over patches of the image.     
- An image is split into fixed-size patches, each of them are then linearly embedded, position embeddings are added, and the resulting sequence of vectors is fed to a standard Transformer encoder.    
- In order to perform classification, the standard approach of adding an extra learnable “classification token” to the sequence is used.   

<br>

## Architecture

<br>

Overall architecture can be described easily in five simple steps:

1. Split an input image into patches
2. Get linear embeddings (representations) from each patch referred to as Patch Embeddings
3. Add positional embeddings and a [CLS] token to each of the Patch Embeddings
there is more to the CLS token that we would cover today. Would request you to consider the definition of CLS token as shared in the last class as wrong, as we need to further decode it.
4. Pass through a Transformer Encoder and get the output values for each of the [CLS] tokens.  
5. Pass the representations of [CLS] tokens through an MLP Head to get final class predictions.

<br>

![ViT](../Results/Session%2017/vit.png)

<br>