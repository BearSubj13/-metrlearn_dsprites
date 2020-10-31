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

#model.load_state_dict(torch.load('contrastive_learning.pt'))
# latent1 = (0, 1, 1, 1, 1, 20)
# index = dspites_dataset.latent_to_index(latent1)
# image = dspites_dataset.__getitem__(index)
# plt.figure()
# plt.imsave("image1.png", image)
# image = torch.FloatTensor([[image]])
# image = image.to(device)
# embedding1 = model.encode(image)
# print(embedding1)
#
#
# latent2 = (0, 0, 1, 1, 1, 1)
# index = dspites_dataset.latent_to_index(latent2)
# image = dspites_dataset.__getitem__(index)
# plt.figure()
# plt.imsave("image2.png", image)
# image = torch.FloatTensor([[image]])
# image = image.to(device)
# embedding2 = model.encode(image)
# print(embedding2)
# print()
# print('delta:', torch.norm(embedding1-embedding2))

#autoencoder
model.load_state_dict(torch.load('weights/autoencoder_bce_loss_latent12.pt'))
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


# model.load_state_dict(torch.load('contrastive_decoder.pt'))
# image1 = train_val['val'].__getitem__(12)['image']
# plt.imsave("original_image.png", image1)
# model.eval()
# image1 = torch.FloatTensor([[image1]])
# #decoded = composed_transform(image=image1[0,:,:].numpy())['image'].squeeze()
# image1 = image1.to(device)
# z, decoded = model(image1)
# decoded = decoded[0].squeeze()
# decoded = decoded.cpu().detach().numpy()
# plt.figure()
# plt.imsave("decoded_image.png", decoded)
# # #np.save("decoded_image.npy", decoded)