#!/usr/bin/env python
# coding: utf-8

# # _Randomly Connected Neural Networks for Self-Supervised Monocular Depth Estimation_ Demo Notebook

# ## Imports

# In[1]:


from matplotlib import pyplot as plt

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import yaml

from loaders import CityScapesDataset

from model import RandomlyConnectedModel

import train
from train.loss import MonodepthLoss


# ## CUDA

# In[2]:


device = torch.device("cuda") \
    if torch.cuda.is_available() \
        else torch.device("cpu")

# ## Hyperparameters

# In[3]:


with open("config.yml") as f:
    model_config = yaml.load(f, Loader=yaml.Loader)

encoder_config = model_config["encoder"]
decoder_config = model_config["decoder"]


# In[4]:


# Dataset parameters
batch_size = 8
validation_samples = 1000
numberof_workers = 0

# Training parameters
numberof_epochs = 1
learning_rate = 1e-4


# ## Dataset

# ### Transforms

# In[5]:


train_transform = transforms.Compose([
    train.transforms.ResizeImage((256, 512)),
    train.transforms.RandomFlip(0.5),
    train.transforms.ToTensor(),
    train.transforms.RandomAugment(0.5, gamma=(0.8, 1.2),
                                   brightness=(0.5, 2.0),
                                   colour=(0.8, 1.2))
])

val_transform = transforms.Compose([
    train.transforms.ResizeImage((256, 512)),
    train.transforms.ToTensor()
])


# ### Datasets

# In[6]:


train_dataset = CityScapesDataset("../datasets/cityscapes/", "train",
                                  train_transform, limit=200)
                                  
val_dataset = CityScapesDataset("../datasets/cityscapes/", "val",
                                val_transform, validation_samples)


# ### Loaders

# In[7]:


train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=numberof_workers)

val_loader = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=numberof_workers)


# ## Train

# In[8]:


# Temporary solution until config code is set up
#model = RandomlyConnectedModel(nodes=5, seed=42).to(device)
model = RandomlyConnectedModel(load_graph="graphs/nodes_5_seed_42").to(device)
#model = RandomlyConnectedModel(encoder_config, decoder_config).to(device)

numberof_parameters = sum(p.numel() for p in model.parameters())
print(f"Model has {numberof_parameters:,} learnable parameters.")

train.train_model(model, train_loader, numberof_epochs, learning_rate,
                  val_loader=val_loader, evaluate_every=1e4, 
                  save_path="trained/", device=device)


# ## Evaluate

# In[ ]:


model.eval()

# ### Run evaluation

# In[ ]:


loss_function = MonodepthLoss()

train.evaluate_model(model, val_loader, loss_function,
                     save_comparison_to="results/",
                     device=device)


# ### Show comparison results

# In[ ]:


image = Image.open("results/comparison.png")
plt.imshow(image)

