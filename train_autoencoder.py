import json
import os
from data_load import Dspites, train_val_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import AutoEncoder
from utils import image_batch_transformation
import albumentations as alb

composed_transform = alb.Compose([alb.GaussianBlur(p=0.9),\
                                  alb.GaussNoise(var_limit=(0.0, 0.07), mean=0.1, p=0.6), \
                                  alb.GridDistortion(p=0.1, num_steps=10, distort_limit=1.0),\
                                  #alb.ElasticTransform(p=1.0, sigma=0.0, alpha=0, alpha_affine=0),\
                                  alb.OpticalDistortion(p=0.5, distort_limit = (-0.1,0.1), shift_limit=(-0.1, 0.1))
                                  ], p=1)

with open("config.json") as json_file:
   conf = json.load(json_file)
dataset_path = os.path.join(conf['data']['dataset_path'], conf['data']['dataset_file'])
device = conf['train']['device']

model = AutoEncoder(in_channels=1, dec_channels=1, latent_size=conf['model']['latent_size'])
model = model.to(device)
model.load_state_dict(torch.load('weights/autoencoder_bce_loss_latent12.pt'))


dspites_dataset = Dspites(dataset_path)
train_val = train_val_split(dspites_dataset)
val_test = train_val_split(train_val['val'], val_split=0.2)

data_loader_train = DataLoader(train_val['train'], batch_size=conf['train']['batch_size'], shuffle=True, num_workers=2)
data_loader_val = DataLoader(val_test['val'], batch_size=200, shuffle=False, num_workers=1)
data_loader_test = DataLoader(val_test['train'], batch_size=200, shuffle=False, num_workers=1)

loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

def autoencoder_step(model, batch, device, loss_function):
    batch = batch['image']
    # if False:
    # perturbed_batch = image_batch_transformation(batch, composed_transform)
    # perturbed_batch = perturbed_batch.to(device)
    # latent_vector, output = model(perturbed_batch)
    batch = batch.unsqueeze(1)
    batch = batch.type(torch.FloatTensor)
    batch = batch.to(device)
    _, output = model(batch)
    loss = loss_function(output, batch)
    return loss

def autoencoder_validation(data_loader_val, model, device, loss_function):
    loss_list = []
    for batch_i, batch in enumerate(data_loader_val):
        loss = autoencoder_step(model, batch, device, loss_function)
        loss_list.append(loss.item())
    mean_loss = sum(loss_list) / len(loss_list)
    return mean_loss


#model.freeze_encoder()
for epoch in range(25):
   if epoch > 10:
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
   validation_loss = autoencoder_validation(data_loader_val, model, device, loss_function)
   if epoch == 0:
       min_validation_loss = validation_loss
   else:
       min_validation_loss = min(min_validation_loss, validation_loss)
   print('epoch {0}, loss: {1:2.5f}, validation: {2:2.5f}'.format(epoch, mean_epoch_loss, validation_loss))
   if min_validation_loss == validation_loss:
       torch.save(model.state_dict(), 'weights/autoencoder_bce_loss_latent12.pt')


model.load_state_dict(torch.load('weights/autoencoder_bce_loss_latent12.pt'))
test_results = autoencoder_validation(data_loader_val, model, device, loss_function)
print('test result: ', test_results)