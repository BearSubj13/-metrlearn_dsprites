import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as alb
import numpy as np

import json
import os

from data_load import DspitesColored, train_val_split
from models import AutoEncoderWithProjector
from metircs_losses import NT_Xent, metric_Recall_top_K
from utils import image_color_batch_transformation


def recall_validation_step(model, batch, augment_transform_list1, augment_transform_list2, device):
    augment_transform1 = np.random.choice(augment_transform_list1)
    augment_transform2 = np.random.choice(augment_transform_list2)
    augment_transform3 = np.random.choice(augment_transform_list1)
    augment_transform4 = np.random.choice(augment_transform_list2)
    batch1 = image_color_batch_transformation(batch, augment_transform1).to(device)
    batch2 = image_color_batch_transformation(batch, augment_transform2).to(device)
    batch3 = image_color_batch_transformation(batch, augment_transform3).to(device)
    batch4 = image_color_batch_transformation(batch, augment_transform4).to(device)
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


def recall_validation(model, data_loader_val, augment_transform_list1, augment_transform_list2, device):
    recall_list = []
    recall_list10 = []
    for batch_i, batch in enumerate(data_loader_val):
        batch = batch['image']
        batch = batch.type(torch.FloatTensor)
        batch = batch.to(device)
        recall_k = recall_validation_step(model, batch, augment_transform_list1, augment_transform_list2, device)
        recall_list.append(recall_k[0])
        recall_list10.append(recall_k[1])
    recall = sum(recall_list) / len(recall_list)
    recall10 = sum(recall_list10) / len(recall_list10)
    return recall, recall10


def sim_clr_training_step(model, batch, transformation1, transformation2, loss_f, device, optimizer):
    augment_transform1 = np.random.choice(transformation1)
    augment_transform2 = np.random.choice(transformation2)
    augmented_batch1 = image_color_batch_transformation(batch, augment_transform1)
    augmented_batch1 = augmented_batch1.to(device)
    augmented_batch2 = image_color_batch_transformation(batch, augment_transform2)
    augmented_batch2 = augmented_batch2.to(device)

    embedding1 = model.encode(augmented_batch1)
    metric_space1 = model.projector(embedding1)
    embedding2 = model.encode(augmented_batch2)
    metric_space2 = model.projector(embedding2)

    loss = loss_f(metric_space1, metric_space2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def contrastive_learning(model, data_loader_train, conf, epoch_number):
    device = conf['train']['device']
    transform1 = [alb.GaussianBlur(p=1),
                  alb.GaussNoise(var_limit=(0.0, 0.07), mean=0.1, p=1),
                  alb.augmentations.transforms.VerticalFlip(p=1),
                  alb.OpticalDistortion(p=1, distort_limit=(-0.2, 0.2), shift_limit=(-0.2, 0.2)),
                  alb.ColorJitter(p=1),
                  alb.Rotate(p=1, limit=20)]

    transform2 = [alb.GridDistortion(p=1, num_steps=10, distort_limit=1.0),
                  alb.HorizontalFlip(p=1),
                  alb.RandomSizedCrop(p=1, min_max_height=[50, 64], height=64, width=64, w2h_ratio=1),
                  alb.ChannelShuffle(p=1),
                  alb.RandomFog(p=1),
                  alb.ToGray(p=1),
                  alb.GridDropout(p=1, ratio=0.4)
                  ]

    loss_f = NT_Xent(batch_size=conf['train']['batch_size'], temperature=1.0, device=conf['train']['device'])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_list = []

    np.random.seed()
    for epoch in range(epoch_number):
        if epoch == 7000:
            for param in optimizer.param_groups:
                param['lr'] = 0.0005
                print('lr: ', param['lr'])
        for batch in data_loader_train:
            batch = batch['image']
            loss = sim_clr_training_step(model, batch, transform1, transform2, loss_f, device, optimizer)
            loss_list.append(loss.item())
        print('{0} loss: {1}'.format(epoch, sum(loss_list)/len(loss_list)))
        if epoch % 10 == 0:
            recall, _ = recall_validation(model, data_loader_train, transform1, transform2, device)
            print('recall: ', recall)


if __name__ == "__main__":
    with open("config.json") as json_file:
        conf = json.load(json_file)
    device = conf['train']['device']

    number_of_colors = 9
    print('number of colors:', number_of_colors)
    dataset_path = os.path.join(conf['data']['dataset_path'], conf['data']['dataset_file'])
    dspites_dataset_colored = DspitesColored(dataset_path, number_of_colors=number_of_colors)

    #dspites_dataset_colored = dspites_dataset_colored.clone_subset()
    print('dataset length', len(dspites_dataset_colored))

    random_state = np.random.randint(0, 1000000)
    print('random_state:', random_state)
    train_val = train_val_split(dspites_dataset_colored, fixed_size_train=1474, random_state=random_state)#val_split=0.998,)
    data_loader_train = DataLoader(train_val['train'], batch_size=conf['train']['batch_size'], shuffle=True,
                                   num_workers=2, drop_last=True)

    model = AutoEncoderWithProjector(in_channels=3, dec_channels=3, latent_size=conf['model']['latent_size'])
    #model.load_state_dict(torch.load('weights/colored1_metric_learning.pth'))
    model = model.to(device)

    contrastive_learning(model, data_loader_train, conf, 8000)
    torch.save(model.state_dict(), 'weights/colored9_2.pth')

