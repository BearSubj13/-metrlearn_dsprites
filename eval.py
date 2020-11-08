from models import AutoEncoder
import torch
import torch.nn as nn
import numpy as np
import json
import os
from torch.utils.data import DataLoader
from data_load import Dspites, train_val_split
import matplotlib.pyplot as plt

device = 'cuda'

with open("config.json") as json_file:
   conf = json.load(json_file)
dataset_path = os.path.join(conf['data']['dataset_path'], conf['data']['dataset_file'])
dspites_dataset = Dspites(dataset_path)
train_val = train_val_split(dspites_dataset)

model = AutoEncoder(in_channels=1, dec_channels=1, latent_size=conf['model']['latent_size'])
model = model.to(device)


load_path = 'weights/my_algorithm_latent_12.pt'
#autoencoder
model.load_state_dict(torch.load(load_path))
image1 = train_val['train'].__getitem__(0)['image']
plt.imsave("images/original_image1.png", image1)
image2 = train_val['train'].__getitem__(1)['image']
plt.imsave("images/original_image2.png", image2)
image3 = train_val['train'].__getitem__(3000)['image']
plt.imsave("images/original_image3.png", image3)


model.eval()
image = torch.FloatTensor([[image1], [image2], [image3]])
image = image.to(device)
z, decoded = model(image)
decoded1 = decoded[0][0].squeeze()
decoded1 = decoded1.cpu().detach().numpy()
plt.figure()
plt.imsave("images/decoded_image1.png", decoded1)
decoded2 = decoded[1][0].squeeze()
decoded2 = decoded2.cpu().detach().numpy()
plt.imsave("images/decoded_image2.png", decoded2)
decoded3 = decoded[2][0].squeeze()
decoded3 = decoded3.cpu().detach().numpy()
plt.imsave("images/decoded_image3.png", decoded3)



image_decoded = torch.FloatTensor([[decoded1], [decoded2], [decoded3]])
image_decoded = image_decoded.to(device)
z2, decoded2 = model(image_decoded)
print(z[0])
print(z2[0])

model_alternetive = AutoEncoder(in_channels=1, dec_channels=1, latent_size=conf['model']['latent_size'])
model_alternetive = model.to(device)
model_alternetive.load_state_dict(torch.load('weights/autoencoder_contrastive_latent12.pt'))
image_decoded3 = model_alternetive.decode(z2)
decoded1 = image_decoded3[0][0].squeeze()
decoded1 = decoded1.cpu().detach().numpy()
plt.imsave("images/decoded_image_alternative1.png", decoded1)