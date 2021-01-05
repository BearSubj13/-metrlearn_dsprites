import torch
import numpy as np

def image_batch_transformation(batch, transformation):
    perturbed = []
    for k in range(batch.shape[0]):
        image_k = batch[k, :, :].cpu().detach().numpy()
        perturbed.append(transformation(image=image_k)['image'])
    perturbed = torch.FloatTensor(perturbed).unsqueeze(1).to(batch.device)
    return perturbed


def image_color_batch_transformation(batch, transformation):
    perturbed = []
    for k in range(batch.shape[0]):
        image_k = batch[k, :, :].cpu().detach().numpy()
        image_k = np.moveaxis(image_k, [0, 1, 2], [2, 0, 1])
        image_k = transformation(image=image_k)['image']
        image_k = np.moveaxis(image_k, [2, 0, 1], [0, 1, 2])
        perturbed.append(image_k)
    perturbed = torch.FloatTensor(perturbed).to(batch.device)
    return perturbed