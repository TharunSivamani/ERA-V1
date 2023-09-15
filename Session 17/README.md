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