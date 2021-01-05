from models import AutoEncoderWithProjector, SimpleNet
import torch
import torch.nn as nn
import json
import os
from torch.utils.data import DataLoader
from data_load import DspitesColored, train_val_split
import copy


def classification_validation(regressor, model, data_loader, latent_range, device):
    precision_list = []
    for batch_i, batch in enumerate(data_loader):
        # if batch_i == 500:
        #     break
        label = batch['latent'][:, latent_range]  # figure type
        label = label.type(torch.LongTensor) - 1
        label = label.to(device)
        batch = batch['image']#.unsqueeze(1)
        batch = batch.type(torch.float32)
        batch = batch.to(device)

        embedding = model.encode(batch)
        embedding = model.projector(embedding)
        embedding = embedding.detach()
        prediction = regressor(embedding)

        prediction = torch.argmax(prediction, dim=1)
        precision = (prediction == label).sum()
        precision = torch.true_divide(precision, prediction.shape[0])

        precision_list.append(precision)
    mean_precision = sum(precision_list) / len(precision_list)
    return mean_precision


def classification(model, regressor, data_loader_train, data_loader_val, data_loader_test,\
               latent_range, device, epoch_number=45):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(regressor.parameters(), lr=0.001)

    model.eval()
    loss_list = []

    for epoch in range(epoch_number):
       loss_list = []
       regressor.train()
       if epoch > 25:
           for param in optimizer.param_groups:
               param['lr'] = max(0.0001, param['lr'] / 1.2)
               print('lr: ', param['lr'])

       for batch_i, batch in enumerate(data_loader_train):
          label = batch['latent'][:, latent_range]  #coordinates
          label = label.type(torch.LongTensor) - 1
          label = label.to(device)
          batch = batch['image']#.unsqueeze(1)
          batch = batch.type(torch.float32)
          batch = batch.to(device)

          embedding = model.encode(batch)
          embedding = model.projector(embedding)
          embedding = embedding.detach()
          prediction = regressor(embedding)

          loss = loss_function(prediction, label)
          loss_list.append(loss.item())

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
       loss = sum(loss_list)/len(loss_list)
       regressor.eval()
       val_error = classification_validation(regressor, model, data_loader_val, \
                                         latent_range, device)
       if epoch == 0:
           min_val_error = val_error
       else:
           min_val_error = min(min_val_error, val_error)
       if min_val_error == val_error:
           best_model = copy.deepcopy(model)
       print('loss: {0:2.4f}, validation error: {1:1.3f}'.format(loss, val_error))

    regressor.eval()
    test_error = classification_validation(regressor, best_model, data_loader_test,
                                       latent_range, device)
    print('test error: {0:1.3f}'.format(test_error))
    return test_error.item()


def classification_interface(model, data_loader_train, data_loader_val, data_loader_test, \
                         latent_size, latent_range, device, epoch_number, number_of_classes=3):
    # 0-color, 1 - shape, 2 - scale (from 0.5 to 1.0), 3,4 - orientation (cos, sin), 5,6 - position (from 0 to 1)
    classificator = SimpleNet(latent_size=latent_size, number_of_classes=number_of_classes)
    classificator.to(device)
    metric = classification(model, classificator, data_loader_train, data_loader_val, data_loader_test, \
                   latent_range, device, epoch_number)
    return metric


def main():
    with open("config.json") as json_file:
        conf = json.load(json_file)

    device = conf['train']['device']

    dataset_path = os.path.join(conf['data']['dataset_path'], conf['data']['dataset_file'])
    dspites_dataset_colored = DspitesColored(dataset_path, number_of_colors=6)
    train_val = train_val_split(dspites_dataset_colored, val_split=0.2)
    data_loader_train = DataLoader(train_val['train'], batch_size=conf['train']['batch_size'], shuffle=True,
                                   num_workers=2, drop_last=True)

    data_loader_train = DataLoader(train_val['train'], batch_size=conf['train']['batch_size'], shuffle=True,
                                   num_workers=2)
    data_loader_val = DataLoader(train_val['val'], batch_size=200, shuffle=False, num_workers=1)
    data_loader_test = DataLoader(train_val['train'], batch_size=200, shuffle=False, num_workers=1)

    print('latent space size:', conf['model']['latent_size'])
    print('batch size:', conf['train']['batch_size'])

    conf['train']['batch_size'] = 128
    data_loader_train = DataLoader(train_val['train'], batch_size=conf['train']['batch_size'], shuffle=True,
                                   num_workers=2)
    data_loader_val = DataLoader(train_val['val'], batch_size=500, shuffle=False, num_workers=1)

    model = AutoEncoderWithProjector(in_channels=3, dec_channels=3, latent_size=conf['model']['latent_size'])
    model = model.to(device)
    #
    # autoencoder_bce_loss_latent12.pt
    number_of_colors = 6
    model.load_state_dict(torch.load('weights/colored6_metric_learning.pth'))

    # 0-color, 1 - shape, 2 - scale (from 0.5 to 1.0), 3,4 - orientation (cos, sin), 5,6 - position (from 0 to 1)
    latent_range = 0
    # min_value = 0.0
    # max_value = 1

    # classificator = SimpleNet(latent_size=conf['model']['latent_size'], number_of_classes=7)
    # classificator.to(device)
    # classification(model, classificator, data_loader_train, data_loader_val, data_loader_test, \
    #            latent_range, device)
    classification_interface(model, data_loader_train, data_loader_val, data_loader_test, \
                             conf['model']['latent_size'], latent_range, device, 35, number_of_colors)


if __name__ == "__main__":
    main()
