from models import AutoEncoder
import torch
import torch.nn as nn
import numpy as np
import json
import os
from torch.utils.data import DataLoader
from data_load import Dspites, train_val_split
import albumentations as alb
from metircs_losses import metric_Recall_top_K
from utils import image_batch_transformation



with open("config.json") as json_file:
   conf = json.load(json_file)
device = conf['train']['device']

dataset_path = os.path.join(conf['data']['dataset_path'], conf['data']['dataset_file'])
dspites_dataset = Dspites(dataset_path)
train_val = train_val_split(dspites_dataset)

print('latent space size:', conf['model']['latent_size'])
print('batch size:', conf['train']['batch_size'])
print('margin:', conf['train']['margin'])

model = AutoEncoder(in_channels=1, dec_channels=1, latent_size=conf['model']['latent_size'])
model = model.to(device)

transform2 = [alb.GaussianBlur(p=1), alb.GaussNoise(var_limit=(0.0, 0.07), mean=0.1, p=1)]
transform1 = [alb.GridDistortion(p=0.2, num_steps=10, distort_limit=1.0), alb.OpticalDistortion(p=1, distort_limit = (-0.1,0.1), shift_limit=(-0.1, 0.1))]
data_loader_train = DataLoader(train_val['train'], batch_size=conf['train']['batch_size'], shuffle=True, num_workers=2)
data_loader_val = DataLoader(train_val['val'], batch_size=100, shuffle=False, num_workers=1)

loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=conf['train']['lr'])
triplet_loss = nn.TripletMarginLoss(margin=conf['train']['margin'], p=2)

model.train()
loss_list = []


# def simCLR_step(batch, augment_transform_list1, augment_transform_list2):
#     augment_transform1 = np.random.choice(augment_transform_list1)
#     augment_transform2 = np.random.choice(augment_transform_list2)
#     batch_1 = []
#     batch_2 = []
#     for k in range(batch.shape[0]):
#         batch_1.append(augment_transform1(image=batch[k, :, :].numpy())['image'])
#         batch_2.append(augment_transform2(image=batch[k, :, :].numpy())['image'])



def triplet_step(model, batch, augment_transform_list1, augment_transform_list2):
    augment_transform1 = np.random.choice(augment_transform_list1)
    augment_transform2 = np.random.choice(augment_transform_list2)
    batch1 = image_batch_transformation(batch, augment_transform1)
    batch2 = image_batch_transformation(batch, augment_transform2)
    anchor = model.encode(batch1)
    positive = model.encode(batch2)

    permutation = torch.randperm(batch.shape[0])
    batch1 = batch1[permutation, :, :]
    permutation = torch.randperm(batch.shape[0])
    batch2 = batch2[permutation, :, :]
    permutation = torch.randperm(batch.shape[0])
    batch3 = batch1[permutation, :, :]
    negative1 = model.encode(batch1)
    negative2 = model.encode(batch2)
    negative3 = model.encode(batch3)

    loss11 = triplet_loss(anchor, positive, negative1)
    loss12 = triplet_loss(anchor, positive, negative2)
    loss13 = triplet_loss(anchor, positive, negative3)
    loss21 = triplet_loss(positive, anchor, negative1)
    loss22 = triplet_loss(positive, anchor, negative2)
    loss23 = triplet_loss(positive, anchor, negative3)
    loss = loss11 + loss12 + loss13 + loss21 + loss22 + loss23
    return loss

def recall_validation_step(model, batch, augment_transform_list1, augment_transform_list2):
        augment_transform1 = np.random.choice(augment_transform_list1)
        augment_transform2 = np.random.choice(augment_transform_list2)
        augment_transform3 = np.random.choice(augment_transform_list1)
        augment_transform4 = np.random.choice(augment_transform_list2)
        batch1 = image_batch_transformation(batch, augment_transform1)
        batch2 = image_batch_transformation(batch, augment_transform2)
        batch3 = image_batch_transformation(batch, augment_transform3)
        batch4 = image_batch_transformation(batch, augment_transform4)
        embedding1 = model.encode(batch1)
        embedding2 = model.encode(batch2)
        embedding3 = model.encode(batch3)
        embedding4 = model.encode(batch4)
        batch_composed = torch.cat((embedding1, embedding2, embedding3, embedding4), dim=0)
        batch_composed = batch_composed.squeeze()

        y_label = torch.arange(0, batch.shape[0]).long()
        y_label = torch.cat((y_label, y_label, y_label, y_label), dim=0)
        y_label = y_label.to(batch.device)

        recall_k = metric_Recall_top_K(batch_composed.squeeze(), y_label, K=3, metric='cosine')
        return recall_k

def recall_validation(model, data_loader_val, augment_transform_list1, augment_transform_list2):
    recall_list = []
    recall_list10 = []
    for batch_i, batch in enumerate(data_loader_val):
        # if batch_i == 100:
        #     break
        batch = batch['image']
        batch = batch.type(torch.FloatTensor)
        batch = batch.to(device)
        recall_k = recall_validation_step(model, batch, augment_transform_list1, augment_transform_list2)
        recall_list.append(recall_k[0])
        recall_list10.append(recall_k[1])
    recall = sum(recall_list) / len(recall_list)
    recall10 = sum(recall_list10) / len(recall_list10)
    return recall, recall10


model.load_state_dict(torch.load('contrastive_learning2.pt'))

for epoch in range(7):
   loss_list = []
   for batch_i, batch in enumerate(data_loader_train):
      # if batch_i == 1000:
      #     break
      batch = batch['image']
      batch = batch.type(torch.FloatTensor)
      batch = batch.to(device)
      loss = triplet_step(model, batch, transform1, transform2)
      loss_list.append(loss.item())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
   print('epoch {0}, loss {1:2.4f}'.format(epoch, sum(loss_list) / len(loss_list) ))
   recall, recall10 = recall_validation(model, data_loader_val, transform1, transform2)
   print('recall@3: {0:2.4f}, recall 10%: {1:2.4f}'.format(recall, recall10))

torch.save(model.state_dict(), 'contrastive_learning2.pt')



