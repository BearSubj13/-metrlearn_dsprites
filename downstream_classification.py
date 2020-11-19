from models import AutoEncoder, SimpleNet, ComplexNet
import torch
import torch.nn as nn
import json
import os
from torch.utils.data import DataLoader
from data_load import Dspites, train_val_split
import copy

with open("config.json") as json_file:
    conf = json.load(json_file)
dataset_path = os.path.join(conf['data']['dataset_path'], conf['data']['dataset_file'])

device = conf['train']['device']

dataset_path = os.path.join(conf['data']['dataset_path'], conf['data']['dataset_file'])
dspites_dataset = Dspites(dataset_path)
train_val = train_val_split(dspites_dataset)
val_test = train_val_split(train_val['val'], val_split=0.2)

data_loader_train = DataLoader(train_val['train'], batch_size=conf['train']['batch_size'], shuffle=True, num_workers=2)
data_loader_val = DataLoader(val_test['val'], batch_size=200, shuffle=False, num_workers=1)
data_loader_test = DataLoader(val_test['train'], batch_size=200, shuffle=False, num_workers=1)

conf['train']['batch_size'] = 128
print('latent space size:', conf['model']['latent_size'])
print('batch size:', conf['train']['batch_size'])

data_loader_train = DataLoader(train_val['train'], batch_size=conf['train']['batch_size'], shuffle=True, num_workers=2)
data_loader_val = DataLoader(train_val['val'], batch_size=500, shuffle=False, num_workers=1)
load_path = 'weights/my_algorithm_2triplet_5.pt'

model = AutoEncoder(in_channels=1, dec_channels=1, latent_size=conf['model']['latent_size'])
model = model.to(device)
model.load_state_dict(torch.load(load_path))

classifier = SimpleNet(conf['model']['latent_size'])
#classifier = ComplexNet(in_channels=1, dec_channels=1, latent_size=conf['model']['latent_size'])
#print(dir(classifier))
classifier.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

def classification_validation(classifier, model, data_loader):
    precision_list = []
    for batch_i, batch in enumerate(data_loader):
        # if batch_i == 500:
        #     break
        label = batch['latent'][:, 0]  # 0 - figure type
        label = label.type(torch.LongTensor) - 1
        label = label.to(device)
        batch = batch['image'].unsqueeze(1)
        batch = batch.type(torch.FloatTensor)
        batch = batch.to(device)

        embedding = model.encode(batch)
        embedding = embedding.detach()
        del batch
        torch.cuda.empty_cache()
        prediction = classifier(embedding)
        prediction = torch.argmax(prediction, dim=1)
        precision = (prediction == label).sum()
        precision = torch.true_divide(precision, prediction.shape[0])
        precision_list.append(precision.item())
    mean_precision = sum(precision_list) / len(precision_list)
    return mean_precision

model.eval()
loss_list = []
for epoch in range(45):
   loss_list = []
   classifier.train()
   if epoch > 25:
       for param in optimizer.param_groups:
           param['lr'] = max(0.0001, param['lr'] / 1.2)
           print('lr: ', param['lr'])
   for batch_i, batch in enumerate(data_loader_train):
      label = batch['latent'][:,0]  #figure type
      label = label.type(torch.LongTensor) - 1

      label = label.to(device)
      batch = batch['image'].unsqueeze(1)
      batch = batch.type(torch.FloatTensor)
      batch = batch.to(device)

      embedding = model.encode(batch)
      embedding = embedding.detach()
      del batch
      torch.cuda.empty_cache()
      prediction = classifier(embedding)

      loss = loss_function(prediction, label)
      loss_list.append(loss.item())

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
   loss = sum(loss_list)/len(loss_list)
   classifier.eval()
   precision = classification_validation(classifier, model, data_loader_val)
   if epoch == 0:
       max_val_precision = precision
   else:
       max_val_precision = max(max_val_precision, precision)
   if max_val_precision == precision:
       best_model = copy.deepcopy(model)
   print('loss: {0:2.3f}, validation precision:{1:1.3f}'.format(loss, precision))

classifier.eval()
precision = classification_validation(classifier, best_model, data_loader_test)
print('test precision:{0:1.3f}'.format(precision))