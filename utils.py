import torch

def image_batch_transformation(batch, transformation):
    perturbed = []
    for k in range(batch.shape[0]):
        image_k = batch[k, :, :].cpu().detach().numpy()
        perturbed.append(transformation(image=image_k)['image'])
    perturbed = torch.FloatTensor(perturbed).unsqueeze(1).to(batch.device)
    return perturbed