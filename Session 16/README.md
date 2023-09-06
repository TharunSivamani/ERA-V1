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