# ERA - V1

## Session 16 - Transformer Architectures and Speeding them up!

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch-Lightning](https://img.shields.io/badge/pytorch_lightning-v2.0.6-red)](https://lightning.ai/docs/pytorch/latest/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-yellow?style=flat&link=https://huggingface.co/)](https://huggingface.co/)

<br>

# Task

1. Pick the "en-fr" dataset from opus_books
2. Remove all English sentences with more than 150 "tokens"
3. Remove all french sentences where len(fench_sentences) > len(english_sentrnce) + 10
4. Train your own transformer (E-D) (do anything you want, use PyTorch, OCP, PS, AMP, etc), but get your loss under 1.8

<br>

# Objective

* The objective of this assignment:

1. Understand the internal structure of transformers, so you can modify at your will
2. Understand the speed up techniques of training and other aspects of transformers

<br>

# Architecture

![arch](encoder.png)

<br>

# Dataset

The Dataset used here is the `Opus Dataset` from `Hugging Face` Datasets

Link - [Dataset](https://huggingface.co/datasets/opus_books)

* Source Language : `English`
* Target Language : `French`

<br>

```python
Length of filtered dataset : 120677
Train DS Size : 108609
  Val DS Size : 12068
Max length of source sentence: 150
Max length of target sentence: 159
Source Vocab Size : 30000
Target Vocab Size : 30000
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