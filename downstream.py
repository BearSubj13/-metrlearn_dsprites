from models import AutoEncoderWithProjector, SimpleNet
import torch
import json
import os
from torch.utils.data import DataLoader
from data_load import DspitesColored, train_val_split
import numpy as np

from downstream_regression_colored import regression_interface
from downstream_classification_colored import classification_interface

def main(input_file_path, number_of_colors=3, evaluate_color=True):
    print('input file:', input_file_path)
    print('number of colors:', number_of_colors)
    with open("config.json") as json_file:
        conf = json.load(json_file)
    device = conf['train']['device']

    model = AutoEncoderWithProjector(in_channels=3, dec_channels=3, latent_size=conf['model']['latent_size'])
    model = model.to(device)
    model.load_state_dict(torch.load(input_file_path))

    dataset_path = os.path.join(conf['data']['dataset_path'], conf['data']['dataset_file'])
    dspites_dataset_colored = DspitesColored(dataset_path, number_of_colors=number_of_colors)
    conf['train']['batch_size'] = 128
    scale_error_list = []
    coordinates_error_list = []
    accuracy_shape_list = []
    accuracy_color_list = []
    number_of_epochs = 35

    for i in range(2):
        print()
        print('iteration', i)
        if i == 0:
            random_state = 666
        else:
            random_state = np.random.randint(100000)
        train_val = train_val_split(dspites_dataset_colored, val_split=0.3, random_state=random_state)
        val_test = train_val_split(train_val['val'], val_split=0.5, random_state=random_state)

        data_loader_train = DataLoader(train_val['train'], batch_size=conf['train']['batch_size'], shuffle=True,
                                       num_workers=2)
        data_loader_val = DataLoader(val_test['train'], batch_size=500, shuffle=False, num_workers=1)
        data_loader_test = DataLoader(val_test['val'], batch_size=200, shuffle=False, num_workers=1)

        print('scale error')
        # 0-color, 1 - shape, 2 - scale (from 0.5 to 1.0), 3,4 - orientation (cos, sin), 5,6 - position (from 0 to 1)
        latent_range = [2]
        min_value = 0.5
        max_value = 1

        scale_error = regression_interface(model, data_loader_train, data_loader_val, data_loader_test, \
                             conf['model']['latent_size'], latent_range, min_value, max_value, device, number_of_epochs)
        scale_error_list.append(scale_error)

        print('coordinates error')
        latent_range = [5,6]
        min_value = 0.0
        max_value = 1
        coordinates_error = regression_interface(model, data_loader_train, data_loader_val, data_loader_test, \
                             conf['model']['latent_size'], latent_range, min_value, max_value,\
                             device, number_of_epochs + 5)
        coordinates_error_list.append(coordinates_error)

        print('shape accuracy')
        latent_range = 1
        accuracy_shape = classification_interface(model, data_loader_train, data_loader_val, data_loader_test, \
                                 conf['model']['latent_size'], latent_range, device, number_of_epochs)
        print(accuracy_shape)
        accuracy_shape_list.append(accuracy_shape)

        if evaluate_color:
            print('color accuracy')
            latent_range = 0
            accuracy_color = classification_interface(model, data_loader_train, data_loader_val, data_loader_test, \
                                     conf['model']['latent_size'], latent_range, device, number_of_epochs, \
                                     number_of_classes=number_of_colors)
            accuracy_color_list.append(accuracy_color)

    mean_scale_error = sum(scale_error_list) / len(coordinates_error_list)
    mean_coordinates_error = sum(coordinates_error_list) / len(coordinates_error_list)
    mean_accuracy_shape = sum(accuracy_shape_list) / len(accuracy_shape_list)
    print()
    print()
    print('scale error: {0:2.4f}'.format(mean_scale_error))
    print(scale_error_list)
    print('coordinates error: {0:2.4f}'.format(mean_coordinates_error))
    print(coordinates_error_list)
    print('shape accuracy: {0:2.4f}: '.format(mean_accuracy_shape))
    print(accuracy_shape_list)
    if evaluate_color:
        mean_accuracy_color = sum(accuracy_color_list) / len(accuracy_color_list)
        print('shape accuracy: {0:2.4f}: '.format(mean_accuracy_color))
        print(accuracy_color_list)


if __name__ == '__main__':
    main('weights/colored8_2.pth', number_of_colors=8, evaluate_color=True)