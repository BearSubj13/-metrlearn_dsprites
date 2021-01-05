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
from train_autoencoder import autoencoder_validation
from metric_learning import recall_validation, transform1, transform2, triplet_loss
import copy

load_path = 'weights/my_algorithm_2triplet_5.pt' #'weights/my_algorithm_2triplet.pt'
save_path = 'weights/my_algorithm_2triplet_6.pt' #'weights/my_algorithm_2triplet_3.pt'

with open("config.json") as json_file:
    conf = json.load(json_file)

#transform2 = [alb.GaussianBlur(p=1), alb.GaussNoise(var_limit=(0.0, 0.07), mean=0.1, p=1)]
#transform1 = [alb.GridDistortion(p=0.2, num_steps=10, distort_limit=1.0), alb.OpticalDistortion(p=1, distort_limit = (-0.1,0.1), shift_limit=(-0.1, 0.1))]

#triplet_loss = nn.TripletMarginLoss(margin=conf['train']['margin'], p=2)

def autoencoder_step(model, batch, device, loss_function):
    batch = batch['image']
    batch = batch.unsqueeze(1)
    batch = batch.type(torch.FloatTensor)
    batch = batch.to(device)
    z, output = model(batch)
    loss = loss_function(output, batch)
    return loss, output, z


def return_loss_function(encoder_decoder):
    def loss_f(x_est, x_gt):
        latent_estim = encoder_decoder.encode(x_est)
        latent_gt = encoder_decoder.encode(x_gt)
        loss = 1 - nn.functional.cosine_similarity(latent_estim, latent_gt)
        loss = loss.mean()
        return loss

    return loss_f

def triplet_step(model, batch, negative, augment_transform_list1, augment_transform_list2):
    batch = batch['image']
    #batch = batch.unsqueeze(1)
    batch = batch.type(torch.FloatTensor)
    batch = batch.to(device)

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
    permutation = torch.randperm(batch.shape[0])
    batch3 = batch2[permutation, :, :]
    negative4 = model.encode(batch3)

    loss11 = triplet_loss(anchor, positive, negative1)
    loss12 = triplet_loss(anchor, positive, negative2)
    loss13 = triplet_loss(anchor, positive, negative3)
    loss14 = triplet_loss(anchor, positive, negative4)
    loss21 = triplet_loss(positive, anchor, negative1)
    loss22 = triplet_loss(positive, anchor, negative2)
    loss23 = triplet_loss(positive, anchor, negative3)
    loss24 = triplet_loss(positive, anchor, negative4)

    negative_z = model.encode(negative)
    penalty = triplet_loss(positive, anchor, negative_z) + triplet_loss(anchor, positive, negative_z)

    loss = loss11 + loss12 + loss13 + loss14 + \
           loss21 + loss22 + loss23 + loss24 + penalty

    return loss, penalty

def decoder_step(model, loss_function, optimizer, data_loader_train, data_loader_val, device):
    loss_list = []
    model.train()

    for batch_i, batch in enumerate(data_loader_train):
        loss, _, _ = autoencoder_step(model, batch, device, loss_function)
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mean_epoch_loss = sum(loss_list) / len(loss_list)
    model.eval()
    validation_loss = autoencoder_validation(data_loader_val, model, device, nn.BCELoss())
    return mean_epoch_loss, validation_loss

def train_decoder(model, data_loader_train, data_loader_val, data_loader_test, device):
    model.train()
    model_frozen = copy.deepcopy(model)
    model_frozen.eval()
    model.freeze_encoder()
    loss_function = return_loss_function(model_frozen)

    for epoch in range(35):
       if epoch > 20:
           for param in optimizer.param_groups:
               param['lr'] = max(0.00001, param['lr'] / conf['train']['lr_decay'])
               print('lr: ', param['lr'])

       mean_epoch_loss, validation_loss = decoder_step(model, loss_function, data_loader_train, data_loader_val, device)

       if epoch == 0:
           min_validation_loss = validation_loss
       else:
           min_validation_loss = min(min_validation_loss, validation_loss)
       print('epoch {0}, loss: {1:2.5f}, validation: {2:2.5f}'.format(epoch, mean_epoch_loss, validation_loss))
       torch.save(model.state_dict(), save_path)

    test_results = autoencoder_validation(data_loader_test, model, device, nn.BCELoss())
    print('test result: ', test_results)

#do not forget to change to model.train()
def image_locality(z, model, margin=conf['train']['margin'], radius=1):
    eps = radius*torch.randn_like(z)
    eps[torch.abs(eps) < 0.5] = 0.5*torch.sign(eps[torch.abs(eps) < 0.5])
    # eps_norm = torch.norm(eps, dim=1)
    # print(eps_norm.shape, eps.shape[1])
    # eps_norm = eps_norm.repeat(eps.shape[1], 1).transpose(1,0)
    # print(eps_norm.shape)
    # eps = torch.div(eps, eps_norm)
    # min_delta = 2.5
    # index_out_of_delta = eps_norm < min_delta
    # eps_norm[index_out_of_delta] = min_delta
    # eps = eps * eps_norm
    z = z + margin * eps
    z = nn.functional.normalize(z, p=2, dim=1)
    decoded = model.decode(z)
    return decoded

def contrastive_autoencoder(model, optimizer, data_loader_train, data_loader_val, device):
    encoder_loss_list = []
    metric_lr_loss_list = []
    penalty_list = []
    model.train()
    for batch_i, batch in enumerate(data_loader_train):
        model_frozen = copy.deepcopy(model)
        model_frozen.eval()
        loss_function = return_loss_function(model_frozen)
        # train decoder, encoder frozen
        model.freeze_encoder()
        loss, _, z = autoencoder_step(model, batch, device, loss_function)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        encoder_loss_list.append(loss.item())
        #random images in close vicinity of original
        z = z.detach()
        model.eval()
        decoded = image_locality(z, model, margin=conf['train']['margin'], radius=1.5)
        decoded = decoded.detach()

        #train encoder by contrastive learning
        model.unfreeze_encoder()
        model.train()
        loss, vicinity_penalty = triplet_step(model, batch, decoded, transform1, transform2)
        metric_lr_loss_list.append(loss.item())
        penalty_list.append(vicinity_penalty.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    validation_loss = autoencoder_validation(data_loader_val, model, device, nn.BCELoss())
    #recall, recall10 = recall_validation(model, data_loader_val, transform1, transform2, device)
    recall, recall10 = 666, 666
    mean_epoch_loss_encoder = sum(encoder_loss_list) / len(encoder_loss_list)
    mean_epoch_loss_metric = sum(metric_lr_loss_list) / len(metric_lr_loss_list)
    mean_penalty = sum(penalty_list) / len(penalty_list)
    model.train()
    return validation_loss, mean_epoch_loss_encoder, mean_penalty, mean_epoch_loss_metric

#
def iterative_training(model, optimizer, data_loader_train, data_loader_val, device):
    for epoch in range(50):
        if epoch > 15:
            for param in optimizer.param_groups:
                param['lr'] = max(0.0001, param['lr'] / conf['train']['lr_decay'])
                print('lr: ', param['lr'])
        validation_loss, mean_epoch_loss_encoder, penalty, mean_epoch_loss_metric \
            = contrastive_autoencoder(model, optimizer, data_loader_train, data_loader_val, device)
        print('epoch {0}, autoencoder loss: {1:2.5f}, BCE val: {2:2.5f}, loss metric learn: {3:1.5f}, penalty: {4:1.5f}'
              .format(epoch, mean_epoch_loss_encoder, validation_loss, mean_epoch_loss_metric, penalty) )

        model_frozen = copy.deepcopy(model)
        model_frozen.eval()
        model.freeze_encoder()
        loss_function = return_loss_function(model_frozen)
        mean_epoch_loss, validation_loss = \
            decoder_step(model, loss_function, optimizer, data_loader_train, data_loader_val, device)
        print('         autoencoder loss: {0:2.5f}, BCE val: {1:2.5f}'.format(mean_epoch_loss, validation_loss))


if __name__ == "__main__":
    device = conf['train']['device']

    model = AutoEncoder(in_channels=1, dec_channels=1, latent_size=conf['model']['latent_size'])
    model = model.to(device)
    model.load_state_dict(torch.load(load_path))

    dataset_path = os.path.join(conf['data']['dataset_path'], conf['data']['dataset_file'])
    dspites_dataset = Dspites(dataset_path)
    train_val = train_val_split(dspites_dataset)
    val_test = train_val_split(train_val['val'], val_split=0.2)

    data_loader_train = DataLoader(train_val['train'], batch_size=conf['train']['batch_size'], shuffle=True,
                                   num_workers=2)
    data_loader_val = DataLoader(val_test['val'], batch_size=200, shuffle=False, num_workers=1)
    data_loader_test = DataLoader(val_test['train'], batch_size=200, shuffle=False, num_workers=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=conf['train']['lr'])

    print('my algorithm')
    print('train dataset length: ', len(train_val['train']))
    print('val dataset length: ', len(val_test['val']))
    print('test dataset length: ', len(val_test['train']))

    print('latent space size:', conf['model']['latent_size'])
    print('batch size:', conf['train']['batch_size'])
    print('margin:', conf['train']['margin'])


    # for epoch in range(60):
    #     if epoch > 15:
    #         for param in optimizer.param_groups:
    #             param['lr'] = max(0.00005, param['lr'] / conf['train']['lr_decay'])
    #             print('lr: ', param['lr'])
    #     validation_loss, recall, recall10, mean_epoch_loss_encoder, penalty, mean_epoch_loss_metric \
    #         = contrastive_autoencoder(model, optimizer, data_loader_train, data_loader_val, device)
    #     print('loss: {0:2.5f}, autoencoder val: {1:2.5f}, loss metric learn: {2:1.5f}, penalty: {3:1.5f}'
    #           .format(mean_epoch_loss_encoder, validation_loss, mean_epoch_loss_metric, penalty) )

    iterative_training(model, optimizer, data_loader_train, data_loader_val, device)

    torch.save(model.state_dict(), save_path)

    print('calculating recall')
    #recall, recall10 = recall_validation(model, data_loader_val, transform1, transform2, device)
    #print('train mode, recall: {0:2.4f}, recall10: {1:2.4f}'.format(recall, recall10))
    model.eval()
    recall, recall10 = recall_validation(model, data_loader_val, transform1, transform2, device)
    print('eval mode, recall: {0:2.4f}, recall10: {1:2.4f}'.format(recall, recall10))