import json
import os
from data_load import Dspites, train_val_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import AutoEncoder
from utils import image_batch_transformation
import albumentations as alb
import numpy as np


if __name__ == "__main__":
    load_path = 'weights/autoencoder_contrastive_latent12.pt'
    save_path = 'weights/autoencoder_contrastive_latent12.pt'
    freeze_encoder = True

    transform2 = [alb.GaussianBlur(p=1), alb.GaussNoise(var_limit=(0.0, 0.07), mean=0.1, p=1)]
    transform1 = [alb.GridDistortion(p=0.2, num_steps=10, distort_limit=1.0),
                  alb.OpticalDistortion(p=1, distort_limit=(-0.1, 0.1), shift_limit=(-0.1, 0.1))]


def autoencoder_step(model, batch, device, loss_function):
    batch = batch['image']
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


def main():
    with open("config.json") as json_file:
        conf = json.load(json_file)
    dataset_path = os.path.join(conf['data']['dataset_path'], conf['data']['dataset_file'])
    device = conf['train']['device']

    model = AutoEncoder(in_channels=1, dec_channels=1, latent_size=conf['model']['latent_size'])
    model = model.to(device)
    model.load_state_dict(torch.load(load_path))

    dspites_dataset = Dspites(dataset_path)
    train_val = train_val_split(dspites_dataset)
    val_test = train_val_split(train_val['val'], val_split=0.2)

    data_loader_train = DataLoader(train_val['train'], batch_size=conf['train']['batch_size'], shuffle=True,
                                   num_workers=2)
    data_loader_val = DataLoader(val_test['val'], batch_size=200, shuffle=False, num_workers=1)
    data_loader_test = DataLoader(val_test['train'], batch_size=200, shuffle=False, num_workers=1)

    print('autoencoder training')
    print('frozen encoder: ', freeze_encoder)
    print('train dataset length: ', len(train_val['train']))
    print('val dataset length: ', len(val_test['val']))
    print('test dataset length: ', len(val_test['train']))

    print('latent space size:', conf['model']['latent_size'])
    print('batch size:', conf['train']['batch_size'])

    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    if freeze_encoder:
        model.freeze_encoder()

    for epoch in range(25):
       if epoch > 15:
           for param in optimizer.param_groups:
               param['lr'] = max(0.00001, param['lr'] / conf['train']['lr_decay'])
               print('lr: ', param['lr'])

       loss_list = []
       model.train()

       for batch_i, batch in enumerate(data_loader_train):
          augment_transform = np.random.choice(augment_transform_list1)
          batch1 = image_batch_transformation(batch, augment_transform)
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
           #pass
           torch.save(model.state_dict(), save_path)


    model.load_state_dict(torch.load(save_path))
    test_results = autoencoder_validation(data_loader_test, model, device, loss_function)
    print('test result: ', test_results)


if __name__ == "__main__":
    main()