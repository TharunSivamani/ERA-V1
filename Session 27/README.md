# Session 27

# PEFT, LoRA, QLoRA, and Fine-tuning LLMs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![trl 0.7.4](https://img.shields.io/badge/trl-v0.7.4-violet)](https://huggingface.co/docs/trl/index)
[![Transformers 4.36.2](https://img.shields.io/badge/transformers-v4.36.2-red)](https://huggingface.co/docs/transformers/index)
[![Accelerate 0.25.0](https://img.shields.io/badge/accelerate-v0.25.0-green)](https://huggingface.co/docs/accelerate/index)
[![PEFT 0.7.1](https://img.shields.io/badge/peft-v0.7.1-lightblue)](https://huggingface.co/docs/peft/index)
[![datasets 2.15.0](https://img.shields.io/badge/datasets-v2.15.0-orange)](https://huggingface.co/docs/datasets/index)
[![bitsandbytes 0.41.3.post2](https://img.shields.io/badge/bitsandbytes-v0.41.3.post2-green)](https://huggingface.co/blog/hf-bitsandbytes-integration)
[![einops 0.7.0](https://img.shields.io/badge/einops-v0.7.0-blue)](https://einops.rocks/)
[![wandb 0.16.1](https://img.shields.io/badge/wandb-v0.16.1-lightgreen)](https://wandb.ai/site)

<br>

# Assignment

- Use this dataset: [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1).   
- Fine-tune the model. It's new, so you'll have to make some changes (including sequence length)!.     
- Use QLoRA strategy.

<br>

# Solution

This repository contains a Jupyter notebook that trained `Microsoft Phi-2` model on `OpenAssistant` dataset.

<br>

# What is PEFT Finetuning?
PEFT Finetuning is Parameter Efficient Fine Tuning, a set of fine-tuning techniques that allows you to fine-tune and train models much more efficiently than normal training. PEFT techniques usually work by reducing the number of trainable parameters in a neural network. The most famous and in-use PEFT techniques are Prefix Tuning, P-tuning, LoRA, etc. LoRA is perhaps the most used one. LoRA also has many variants like QLoRA and LongLoRA, which have their own applications.


# LoRA Finetuning
LoRA is the most popular and perhaps the most used PEFT technique, but was released back in 2021 in [this](https://arxiv.org/abs/2012.13255) paper. LoRA is more of an adapter approach, where it introduces new parameters into the model to train the model through these new parameters. The trick is in how the new params are introduced and merged back into the model, without increasing the total number of params in the model.


# LoRA finetuning with HuggingFace
To implement LoRA finetuning with HuggingFace, you need to use the [PEFT library](https://pypi.org/project/peft/) to inject the LoRA adapters into the model and use them as the update matrices.

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=True) # load the model

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=32, lora_alpha=16, lora_dropout=0.1,
    target_modules=['query_key_value'] # optional, you can target specific layers using this
) # create LoRA config for the finetuning

model = get_peft_model(model, peft_config) # create a model ready for LoRA finetuning

model.print_trainable_parameters() 
# trainable params: 9,437,184 || all params: 6,931,162,432 || trainable%: 0.13615586263611604
```

# How is QLoRA different from LoRA?
QLoRA and LoRA both are finetuning techniques, but QLoRA uses LoRA as an accessory to fix the errors introduced during the quantization errors. LoRA in itself is more of a standalone finetuning technique.

# QLoRA Finetuning
QLoRA is a finetuning technique that combines a high-precision computing technique with a low-precision storage method. This helps keep the model size small while still making sure the model is still highly performant and accurate.

# QLoRA finetuning with HuggingFace
To do QLoRA finetuning with HuggingFace, you need to install both the [BitsandBytes](https://pypi.org/project/bitsandbytes/) library and the PEFT library. The BitsandBytes library takes care of the 4-bit quantization and the whole low-precision storage and high-precision compute part. The PEFT library will be used for the LoRA finetuning part.


```python
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
model_id = "EleutherAI/gpt-neox-20b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
) # setup bits and bytes config

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model) # prepares the whole model for kbit training

config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["query_key_value"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config) # Now you get a model ready for QLoRA training
```