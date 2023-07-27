import os
import torch
from math import sqrt, floor, ceil
from typing import Iterable, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
import torchinfo

SEED = 42

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    if get_device() == 'cuda':
        torch.cuda.manual_seed(seed)

def get_device() -> tuple:

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    return device


def denormalize(img):
    channel_means = (0.4914, 0.4822, 0.4465)
    channel_stdevs = (0.2470, 0.2435, 0.2616)
    img = img.astype(dtype=np.float32)

    for i in range(img.shape[0]):
        img[i] = (img[i] * channel_stdevs[i]) + channel_means[i]

    return np.transpose(img, (1, 2, 0))

def get_rows_cols(num: int) -> Tuple[int, int]:
    cols = floor(sqrt(num))
    rows = ceil(num / cols)

    return rows, cols

def plot_examples(images, labels, figsize=(15,5)):
    _ = plt.figure(figsize=figsize)

    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.tight_layout()
        image = images[i]
        plt.imshow(image, cmap='gray')
        label = labels[i]
        plt.title(str(label))
        plt.xticks([])
        plt.yticks([])

def visualize_data(
    loader,
    num_figures: int = 12,
    label: str = "",
    classes: List[str] = [],
):

    batch_data, batch_label = next(iter(loader))

    fig = plt.figure()
    fig.suptitle(label)

    rows, cols = get_rows_cols(num_figures)

    for i in range(num_figures):
        plt.subplot(rows, cols, i + 1)
        plt.tight_layout()
        npimg = denormalize(batch_data[i].cpu().numpy().squeeze())
        label = (
            classes[batch_label[i]] if batch_label[i] < len(classes) else batch_label[i]
        )
        plt.imshow(npimg, cmap="gray")
        plt.title(label)
        plt.xticks([])
        plt.yticks([])

def model_summary(model, input_size=None):
    return torchinfo.summary(model, input_size=input_size, depth=5,
                             col_names=["input_size", "output_size", "num_params", "params_percent"])