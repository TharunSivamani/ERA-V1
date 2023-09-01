# ERA - V1

## Session 15 - Dawn of Transformers - Part II Part A & B

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch-Lightning](https://img.shields.io/badge/pytorch_lightning-v2.0.6-red)](https://lightning.ai/docs/pytorch/latest/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-yellow?style=flat&link=https://huggingface.co/)](https://huggingface.co/)

<br>

# Task

1. Rewrite the whole code covered in the class in Pytorch-Lightning (code copy will not be provided)
2. Train the model for 10 epochs
3. Achieve a loss of less than 4

<br>

# Objective

* The objective of this assignment:

1. Understand the internal structure of transformers, so you can modify at your will
2. Loss should start from 9-10, and reduce to 4, showing that your code is working

<br>

# Architecture

![arch](../Results/Session%2015/encoder.png)

<br>

# Dataset

The Dataset used here is the `Opus Dataset` from `Hugging Face` Datasets

Link - [Dataset](https://huggingface.co/datasets/opus_books)

* Source Language : `English`
* Target Language : `Italian`

<br>

```python
Max length of source sentence: 309
Max length of target sentence: 274
Source Tokenizer Vocab Size : 15698
Target Tokenizer Vocab Size : 22463
```

<br>

# Metrics Used

## Char Error Rate (CER)

* Character Error Rate (CER) is a metric of the performance of an automatic speech recognition (ASR) system.

* This value indicates the percentage of characters that were incorrectly predicted. The lower the value, the better the performance of the ASR system with a CharErrorRate of 0 being a perfect score. Character error rate can then be computed as: 

* $CharErrorRate = \frac{S + D + I}{N} = \frac{S + D + I}{S + D + C}$`

## Word Error Rate

* Word error rate (WordErrorRate) is a common metric of the performance of an automatic speech recognition.

* This value indicates the percentage of words that were incorrectly predicted. The lower the value, the better the performance of the ASR system with a WER of 0 being a perfect score. Word error rate can then be computed as:

* $WER = \frac{S + D + I}{N} = \frac{S + D + I}{S + D + C}$

## BLEU Score

Calculate BLEU score of machine translated text with one or more references.

* As input to forward and update the metric accepts the following input:

    preds (Sequence): An iterable of machine translated corpus

    target (Sequence): An iterable of iterables of reference corpus

As output of forward and update the metric returns the following output:

* bleu (Tensor): A tensor with the BLEU Score

<br>

# Model Summary

```python
  | Name            | Type             | Params
-----------------------------------------------------
0 | net             | Transformer      | 75.1 M
1 | loss_fn         | CrossEntropyLoss | 0     
2 | char_error_rate | _CharErrorRate   | 0     
3 | word_error_rate | _WordErrorRate   | 0     
4 | bleu_score      | _BLEUScore       | 0     
-----------------------------------------------------
75.1 M    Trainable params
0         Non-trainable params
75.1 M    Total params
300.532   Total estimated model params size (MB)
```

<br>

# Inference

![inf](../Results/Session%2015/inference.png)

<br>