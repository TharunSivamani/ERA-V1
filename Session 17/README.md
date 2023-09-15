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

# ViT Model Architecture

<br>

```python
============================================================================================================================================
Layer (type (var_name))                                      Input Shape          Output Shape         Param #              Trainable
============================================================================================================================================
Transformer (Transformer)                                    [32, 3, 224, 224]    [32, 3]              152,064              True
├─PatchEmbedding (patch_embedding)                           [32, 3, 224, 224]    [32, 196, 768]       --                   True
│    └─Conv2d (patcher)                                      [32, 3, 224, 224]    [32, 768, 14, 14]    590,592              True
│    └─Flatten (flatten)                                     [32, 768, 14, 14]    [32, 768, 196]       --                   --
├─Dropout (embedding_dropout)                                [32, 197, 768]       [32, 197, 768]       --                   --
├─Sequential (transformer_encoder)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    └─TransformerEncoderBlock (0)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (1)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (2)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (3)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (4)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (5)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (6)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (7)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (8)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (9)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (10)                          [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (11)                          [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
├─Sequential (classifier)                                    [32, 768]            [32, 3]              --                   True
│    └─LayerNorm (0)                                         [32, 768]            [32, 768]            1,536                True
│    └─Linear (1)                                            [32, 768]            [32, 3]              2,307                True
============================================================================================================================================
Total params: 85,800,963
Trainable params: 85,800,963
Non-trainable params: 0
Total mult-adds (G): 5.52
============================================================================================================================================
Input size (MB): 19.27
Forward/backward pass size (MB): 3292.20
Params size (MB): 229.20
Estimated Total Size (MB): 3540.67
============================================================================================================================================
```

<br>

# Training Logs

<br>

## BERT

<br>

```python
it: 9900  | loss 4.26  | Δw: 10.778
it: 9910  | loss 4.2   | Δw: 11.104
it: 9920  | loss 4.3   | Δw: 10.65
it: 9930  | loss 4.3   | Δw: 11.046
it: 9940  | loss 4.24  | Δw: 11.006
it: 9950  | loss 4.24  | Δw: 10.883
it: 9960  | loss 4.23  | Δw: 10.773
it: 9970  | loss 4.27  | Δw: 10.874
it: 9980  | loss 4.31  | Δw: 10.907
it: 9990  | loss 4.21  | Δw: 11.383
```

<br>

## GPT

<br>

```python
step          0 | train loss 10.6856 | val loss 10.6957
step        500 | train loss 0.5269  | val loss 8.0400
step       1000 | train loss 0.1659  | val loss 9.3979
step       1500 | train loss 0.1406  | val loss 10.1811
step       2000 | train loss 0.1274  | val loss 10.2829
step       2500 | train loss 0.1269  | val loss 10.6556
step       3000 | train loss 0.1191  | val loss 10.5359
step       3500 | train loss 0.1131  | val loss 10.9533
step       4000 | train loss 0.1119  | val loss 10.7073
step       4500 | train loss 0.1064  | val loss 11.1684
step       4999 | train loss 0.1006  | val loss 10.9910
```

<br>

## ViT

```python
Epoch: 1 | train_loss: 3.5604 | train_acc: 0.2891 | test_loss: 2.9965 | test_acc: 0.5417
Epoch: 2 | train_loss: 2.1893 | train_acc: 0.2734 | test_loss: 2.7851 | test_acc: 0.2604
Epoch: 3 | train_loss: 1.4335 | train_acc: 0.4180 | test_loss: 1.3895 | test_acc: 0.2604
Epoch: 4 | train_loss: 1.1844 | train_acc: 0.2891 | test_loss: 1.1820 | test_acc: 0.1979
Epoch: 5 | train_loss: 1.1453 | train_acc: 0.3828 | test_loss: 1.7617 | test_acc: 0.1979
Epoch: 6 | train_loss: 1.1861 | train_acc: 0.3320 | test_loss: 1.2080 | test_acc: 0.1979
Epoch: 7 | train_loss: 1.1100 | train_acc: 0.4219 | test_loss: 1.4391 | test_acc: 0.1979
Epoch: 8 | train_loss: 1.1615 | train_acc: 0.2695 | test_loss: 1.0953 | test_acc: 0.2604
Epoch: 9 | train_loss: 1.1251 | train_acc: 0.2969 | test_loss: 1.2196 | test_acc: 0.1979
Epoch: 10 | train_loss: 1.2304 | train_acc: 0.2812 | test_loss: 1.1582 | test_acc: 0.2604
```
<br>

# Results - ViT

<br>

![ViT results](../Results/Session%2017/vit_graph.png)