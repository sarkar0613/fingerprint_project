import torch
import random
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


"""
input shape: (B, H, W)
output shape: (B, 3, H, W)
"""
def expand_to_3_channels(data):
    data = data.unsqueeze(1)
    return data.repeat(1, 3, 1, 1)


def calculate_mean_std(tensor):
    tensor = tensor.float() / 255.0
    mean = tensor.mean().item()
    std = tensor.std().item()
    return mean, std

def calculate_mean_std_multichannel(tensor):
    tensor = tensor.float() / 255.0
    mean = tensor.mean(dim=(0, 2, 3))
    std = tensor.std(dim=(0, 2, 3))   
    return mean, std

def random_sample_and_show(images, labels, sample_size=5, title="Random Sample"):
    indices = random.sample(range(len(images)), sample_size)
    sampled_images = images[indices].permute(0, 2, 3, 1).cpu().numpy()  
    sampled_labels = labels[indices]

    fig, axes = plt.subplots(1, sample_size, figsize=(20, 5))
    for i, (img, label) in enumerate(zip(sampled_images, sampled_labels)):
        axes[i].imshow(img.astype('uint8')) 
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    plt.suptitle(title)
    plt.show()

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    