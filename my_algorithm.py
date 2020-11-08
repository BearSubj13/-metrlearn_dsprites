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
from train_autoencoder import autoencoder_validation, autoencoder_step
#from metric_learning import recall_validation
import copy

load_path = 'weights/contrastive_learning_latent12.pt'
save_path = 'weights/my_algorithm_latent_12.pt'

with open("config.json") as json_file:
   conf = json.load(json_file)
device = conf['train']['device']

model = AutoEncoder(in_channels=1, dec_channels=1, latent_size=conf['model']['latent_size'])
model = model.to(device)
model.load_state_dict(torch.load(load_path))

dataset_path = os.path.join(conf['data']['dataset_path'], conf['data']['dataset_file'])
dspites_dataset = Dspites(dataset_path)
train_val = train_val_split(dspites_dataset)
val_test = train_val_split(train_val['val'], val_split=0.2)


data_loader_train = DataLoader(train_val['train'], batch_size=conf['train']['batch_size'], shuffle=True, num_workers=2)
data_loader_val = DataLoader(val_test['val'], batch_size=200, shuffle=False, num_workers=1)
data_loader_test = DataLoader(val_test['train'], batch_size=200, shuffle=False, num_workers=1)

print('my algorithm')
print('train dataset length: ', len(train_val['train']))
print('val dataset length: ', len(val_test['val']))
print('test dataset length: ', len(val_test['train']))

print('latent space size:', conf['model']['latent_size'])
print('batch size:', conf['train']['batch_size'])
print('margin:', conf['train']['margin'])

transform2 = [alb.GaussianBlur(p=1), alb.GaussNoise(var_limit=(0.0, 0.07), mean=0.1, p=1)]
transform1 = [alb.GridDistortion(p=0.2, num_steps=10, distort_limit=1.0), alb.OpticalDistortion(p=1, distort_limit = (-0.1,0.1), shift_limit=(-0.1, 0.1))]

triplet_loss = nn.TripletMarginLoss(margin=conf['train']['margin'], p=2)


def return_loss_function(encoder_decoder):
    def loss_f(x_est, x_gt):
        latent_estim = encoder_decoder.encode(x_est)
        latent_gt = encoder_decoder.encode(x_gt)
        loss = 1 - nn.functional.cosine_similarity(latent_estim, latent_gt)
        loss = loss.mean()
        return loss

    return loss_f

def triplet_step(model, batch, negative, augment_transform_list1, augment_transform_list2):
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
    negative1 = model.encode(batch1)
    negative2 = model.encode(batch2)
    negative3 = model.encode(negative)

    loss11 = triplet_loss(anchor, positive, negative1)
    loss12 = triplet_loss(anchor, positive, negative2)
    loss13 = triplet_loss(anchor, positive, negative3)
    loss21 = triplet_loss(positive, anchor, negative1)
    loss22 = triplet_loss(positive, anchor, negative2)
    loss23 = triplet_loss(positive, anchor, negative3)
    loss = loss11 + loss12 + loss13 + loss21 + loss22 + loss23
    return loss


def train_decoder(model, data_loader_train, data_loader_val, data_loader_test, device):
    model.train()
    model_frozen = copy.deepcopy(model)
    model_frozen.eval()
    model.freeze_encoder()
    loss_function = return_loss_function(model_frozen)

    optimizer = torch.optim.Adam(model.parameters(), lr=conf['train']['lr'])

    for epoch in range(25):
       if epoch > 15:
           for param in optimizer.param_groups:
               param['lr'] = max(0.00001, param['lr'] / conf['train']['lr_decay'])
               print('lr: ', param['lr'])

       loss_list = []
       model.train()

       for batch_i, batch in enumerate(data_loader_train):
          loss = autoencoder_step(model, batch, device, loss_function)
          loss_list.append(loss.item())
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()


       mean_epoch_loss = sum(loss_list) / len(loss_list)
       model.eval()
       validation_loss = autoencoder_validation(data_loader_val, model, device, nn.BCELoss())
       if epoch == 0:
           min_validation_loss = validation_loss
       else:
           min_validation_loss = min(min_validation_loss, validation_loss)
       print('epoch {0}, loss: {1:2.5f}, validation: {2:2.5f}'.format(epoch, mean_epoch_loss, validation_loss))
       torch.save(model.state_dict(), save_path)

    test_results = autoencoder_validation(data_loader_test, model, device, nn.BCELoss())
    print('test result: ', test_results)

if __name__ == "__main__":
    train_decoder(model, data_loader_train, data_loader_val, data_loader_test, device)