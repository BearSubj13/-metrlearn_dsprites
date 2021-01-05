from copy import deepcopy
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import math

class Dspites(Dataset):
    def __init__(self, dataset_path):
        #super().__init__(self)
        # modify the default parameters of np.load
        np_load_old = np.load
        load = lambda *a, **k: np_load_old(*a, allow_pickle=True, encoding='latin1', **k)
        # Load dataset
        dataset_zip = load(dataset_path)

        #print('Keys in the dataset:', dataset_zip.keys())
        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        metadata = dataset_zip['metadata'][()]
        self.latents_sizes = metadata['latents_sizes']
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                        np.array([1, ])))

    # 0 Color: white
    # 1 Shape: square, ellipse, heart
    # 2 Scale: 6  values linearly spaced in [0.5, 1]
    # 3 Orientation: 40 values in [0, 2pi]
    # 4 Position  X: 32 values in [0, 1]
    # 5 Position  Y: 32 values in [0, 1]
    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def sample_latent(self, size=1):
        samples = np.zeros((size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)

        return samples

    def __getitem__(self, idx):
        latent_values = self.latents_values[idx]
        angle = latent_values[3]
        for i in range(4):
            angle = angle if angle <= math.pi/2 else angle - math.pi/2
        angle = 4*angle
        new_latent_values = np.array([latent_values[1], latent_values[2], math.cos(angle), \
                                      math.sin(angle), latent_values[4], latent_values[5]])
        return {'image': self.imgs[idx], 'latent': new_latent_values}

    def __len__(self):
        return len(self.imgs)


def train_val_split(dataset, val_split=0.9, fixed_size_train=None, random_state=666):
    if fixed_size_train:
        assert fixed_size_train <= len(dataset)
        val_split = 1 - fixed_size_train / len(dataset)
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, random_state=random_state)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


class DspitesColored(Dataset):
    def __init__(self, dataset_path, number_of_colors=1):
        #super().__init__(self)
        # modify the default parameters of np.load
        np_load_old = np.load
        load = lambda *a, **k: np_load_old(*a, allow_pickle=True, encoding='latin1', **k)
        # Load dataset
        dataset_zip = load(dataset_path)

        #print('Keys in the dataset:', dataset_zip.keys())
        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        metadata = dataset_zip['metadata'][()]
        self.latents_sizes = metadata['latents_sizes']
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                        np.array([1, ])))
        np.random.seed(0)
        self.number_of_colors = number_of_colors
        self.colors_index = np.random.randint(low=0, high=number_of_colors, size=len(self.imgs))
        self.colors = np.array([[1.0, 1.0, 1.0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0],\
                                [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], \
                                [1.0, 0, 0.5], [1.0, 0, 0.5], [0, 1.0, 0.5]])



    # 0 Color: white
    # 1 Shape: square, ellipse, heart
    # 2 Scale: 6  values linearly spaced in [0.5, 1]
    # 3 Orientation: 40 values in [0, 2pi]
    # 4 Position  X: 32 values in [0, 1]
    # 5 Position  Y: 32 values in [0, 1]
    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def sample_latent(self, size=1):
        samples = np.zeros((size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)

        return samples

    def clone_subset(self):
        new_images = []
        new_latent = []
        for i, latent in enumerate(self.latents_values):
            if latent[2] == 0.7:
                new_images.append(self.imgs[i])
                new_latent.append(latent)

        new_dataset = deepcopy(self)
        new_dataset.imgs = deepcopy(new_images)
        new_dataset.latents_values = deepcopy(new_latent)
        return new_dataset

    def __getitem__(self, idx):
        latent_values = self.latents_values[idx]
        angle = latent_values[3]
        for i in range(4):
            angle = angle if angle <= math.pi/2 else angle - math.pi/2
        angle = 4*angle
        new_latent_values = np.array([self.colors_index[idx]+1, latent_values[1], latent_values[2],
                                      math.cos(angle), math.sin(angle), latent_values[4], latent_values[5]])
        image = self.imgs[idx]
        image = image[np.newaxis,:,:].repeat(3,0).astype(np.float32)
        color = self.colors[self.colors_index[idx]]
        for i in range(3):
            image[i,:,:] = image[i,:,:]*color[i]
        return {'image': image, 'latent': new_latent_values}

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    pass
    import json
    import os
    with open("config.json") as json_file:
        conf = json.load(json_file)
    device = conf['train']['device']

    dataset_path = os.path.join(conf['data']['dataset_path'], conf['data']['dataset_file'])
    dspites_dataset = DspitesColored(dataset_path, number_of_colors=7)

    dspiste07scale = dspites_dataset.clone_subset()
    print(len(dspiste07scale))
    print(dspiste07scale.__getitem__(121029)['latent'])
    image = dspiste07scale.__getitem__(121029)['image']
    image = np.moveaxis(image, [0,1,2], [2,0,1])
    image = image.astype(np.float32)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imsave("images/colored.png", image)