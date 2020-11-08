import json
import os
from data_load import Dspites, train_val_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import AutoEncoder
import torch.nn.functional as F


with open("config.json") as json_file:
   conf = json.load(json_file)
dataset_path = os.path.join(conf['data']['dataset_path'], conf['data']['dataset_file'])
device = conf['train']['device']

model = AutoEncoder(in_channels=1, dec_channels=1, latent_size=conf['model']['latent_size'])
model = model.to(device)

dspites_dataset = Dspites(dataset_path)
train_val = train_val_split(dspites_dataset)
val_test = train_val_split(train_val['val'], val_split=0.2)

data_loader_train = DataLoader(train_val['train'], batch_size=conf['train']['batch_size'], shuffle=True, num_workers=2)
data_loader_val = DataLoader(val_test['val'], batch_size=200, shuffle=False, num_workers=1)
data_loader_test = DataLoader(val_test['train'], batch_size=200, shuffle=False, num_workers=1)

loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

model.load_state_dict(torch.load('weights/autoencoder_bce_loss_latent12.pt'))

def autoencoder_step(model, original_batch, device, loss_function):
    original = original_batch['image']
    original = original.unsqueeze(1)
    original = original.type(torch.FloatTensor)
    original = original.to(device)
    with torch.no_grad():
        _, decoded = model(original)
    z_decoded, decoded_decoded = model(decoded.detach())
    z_original, decoded_original = model(original)
    reconstruction_loss_original = loss_function(decoded_original, original)
    reconstruction_loss_decoded = loss_function(decoded_decoded, original)
    reconstruction_loss = reconstruction_loss_original + reconstruction_loss_decoded
    cos_margin = 0.7*torch.ones(z_decoded.shape[0]).to(device)
    embedding_loss = torch.max(cos_margin, F.cosine_similarity(z_decoded, z_original)).mean()
    embedding_loss = 1 - embedding_loss
    delta_coeff = 7*reconstruction_loss_original.detach()
    loss = reconstruction_loss #- (delta_coeff**2)*embedding_loss
    return loss, embedding_loss, reconstruction_loss_original, reconstruction_loss


def autoencoder_validation(data_loader_val, model, device, loss_function):
    loss_list = []
    for batch_i, batch in enumerate(data_loader_val):
        loss, emb_loss, reconstruction_loss_original, reconstr_loss = autoencoder_step(model, batch, device, loss_function)
        loss_list.append(reconstruction_loss_original.item())
    mean_loss = sum(loss_list) / len(loss_list)
    return mean_loss


#model.freeze_encoder()
for epoch in range(35):
   if epoch > 15:
       for param in optimizer.param_groups:
           param['lr'] = max(0.00003, param['lr'] / conf['train']['lr_decay'])
           print('lr: ', param['lr'])

   loss_list = []
   emb_loss_list = []
   delta_loss_list = []
   reconstr_loss_list = []

   model.train()

   for batch_i, batch in enumerate(data_loader_train):
      loss, emb_loss, delta_coeff, reconstr_loss = autoencoder_step(model, batch, device, loss_function)
      loss_list.append(loss.item())
      emb_loss_list.append(emb_loss.item())
      delta_loss_list.append(delta_coeff.item())
      reconstr_loss_list.append(reconstr_loss.item())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

   mean_epoch_loss = sum(loss_list) / len(loss_list)
   mean_ebedding_loss = sum(emb_loss_list) / len(emb_loss_list)
   mean_delta_loss = sum(delta_loss_list) / len(delta_loss_list)
   mean_reconstr_loss = sum(reconstr_loss_list) / len(reconstr_loss_list)
   model.eval()
   validation_loss = autoencoder_validation(data_loader_val, model, device, loss_function)
   if epoch == 0:
       min_validation_loss = validation_loss
   else:
       min_validation_loss = min(min_validation_loss, validation_loss)
   print('epoch {0}, loss: {1:2.5f}, emb loss: {2:2.5f}, reconstr original: {3:2.5f}, validation: {4:2.5f}, reconstract: {5:2.5f}'\
         .format(epoch, mean_epoch_loss, mean_ebedding_loss, mean_delta_loss, validation_loss, mean_reconstr_loss))
   if min_validation_loss == validation_loss:
       torch.save(model.state_dict(), 'weights/autoencoder_enhanced_latent12.pt')

model.load_state_dict(torch.load('weights/autoencoder_enhanced_latent12.pt'))
test_results = autoencoder_validation(data_loader_val, model, device, loss_function)
print('test result: ', test_results)