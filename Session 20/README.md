# Session 20 

# Generative Art and Stable Diffusion

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![Torchvision 0.15](https://img.shields.io/badge/torchvision-v0.15-green)](https://pytorch.org/vision/stable/index.html)
[![Albumentations](https://img.shields.io/badge/albumentations-v1.3.1-yellow)](https://albumentations.ai/docs/)
[![Transformers](https://img.shields.io/badge/transformers-v4.34.0-lightgreen)](https://huggingface.co/docs/transformers/index)
[![Diffusers](https://img.shields.io/badge/diffusers-v0.21.4-darkgreen)](https://huggingface.co/docs/diffusers/index)
[![ftfy](https://img.shields.io/badge/ftfy-v6.1.1-red)](https://ftfy.readthedocs.io/en/latest/)
[![accelerate](https://img.shields.io/badge/accelerate-v0.24.0-lightorange)](https://huggingface.co/docs/accelerate/index)

<br>

# Task

Read the textual inversion section on [this](https://github.com/fastai/diffusion-nbs/blob/master/Stable%20Diffusion%20Deep%20Dive.ipynb) to an external site. page. There's a mention of a **community-created SD concepts library** and a download of the learned_embeds.bin file. There is also a mention of "blue_loss" in the Guidance Section.

- Select 5 different styles of your choice and show output for the same prompt using these 5 different styles. Remember the seeds as you'll use them later. Keep seeds different for each 5 types.
- Now apply your own variant of "blue_loss" (it cannot be red, green, or blue loss) on the same prompts with each concept library and store the results. 

<br>

# Solution

* This repository contains a notebook that demonstrates Stable Diffusion and various concepts like Textual Inversion and some extra control to the generation process.

<br>

# Stable Diffusion Styles

1. Birb style `<birb-style>`
2. Matrix `<hatman-matrix>`
3. egorey `<gorey>`
4. PJablonski style `<pjablonski-style>`
5. Fairy Tale Painting Style `<fairy-tale-painting-style>`
6. Dreamy Painting `<dreamy-painting>`

<br>

## Prompt : 'a cat in a birthday hat'

![sd-styles-output](../Results/Session%2020/style_outputs.png)

<br>

# Blue Loss

* This loss function measures the average absolute difference between the blue channel values of the input images and the target value of 0.9. 
* The lower the value of this loss, the closer the blue channel values are to 0.9, indicating a better match with the desired condition.

```python
def blue_loss(images):
    # How far are the blue channel values to 0.9:
    error = torch.abs(images[:,2] - 0.9).mean() # [:,2] -> all images in batch, only the blue channel
    return error
```

<br>

## Prompt : 'A campfire (oil on canvas)'

![blue-loss](../Results/Session%2020/blue_loss_output.png)

<br>

# Custom Loss

* This custom loss function quantifies the colorfulness of an image batch and encourages the model to generate images with vibrant and diverse colors. 
* Higher values of the colorfulness metric (and consequently, the loss) indicate more colorful images, while lower values indicate less colorful images.

```python
# colorfulness_loss 
def custom_loss(image):
    # Calculate colorfulness metric (standard deviation of RGB channels)
    std_dev = torch.std(image, dim=(1, 2))
    loss = torch.mean(std_dev)
    return loss
```

<br>

## Prompt : "A campfire (oil on canvas)"

![custom-loss](../Results/Session%2020/custom_loss_output.png)

<br>