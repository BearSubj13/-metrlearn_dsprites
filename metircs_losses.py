import torch
import torch.nn as nn
import numpy as np

def outer_pairwise_distance(A, B=None):
    """
        Compute pairwise_distance of Tensors
            A (size(A) = n x d, where n - rows count, d - vector size) and
            B (size(A) = m x d, where m - rows count, d - vector size)
        return matrix C (size n x m), such as C_ij = distance(i-th row matrix A, j-th row matrix B)
        if only one Tensor was given, computer pairwise distance to itself (B = A)
    """

    if B is None: B = A

    max_size = 2 ** 26
    n = A.size(0)
    m = B.size(0)
    d = A.size(1)

    if n * m * d <= max_size or m == 1:

        return torch.pairwise_distance(
            A[:, None].expand(n, m, d).reshape((-1, d)),
            B.expand(n, m, d).reshape((-1, d))
        ).reshape((n, m))

    else:

        batch_size = max(1, max_size // (n * d))
        batch_results = []
        for i in range((m - 1) // batch_size + 1):
            id_left = i * batch_size
            id_rigth = min((i + 1) * batch_size, m)
            batch_results.append(outer_pairwise_distance(A, B[id_left:id_rigth]))

        return torch.cat(batch_results, dim=1)


def outer_cosine_similarity(A, B=None):
    """
        Compute cosine_similarity of Tensors
            A (size(A) = n x d, where n - rows count, d - vector size) and
            B (size(A) = m x d, where m - rows count, d - vector size)
        return matrix C (size n x m), such as C_ij = cosine_similarity(i-th row matrix A, j-th row matrix B)
        if only one Tensor was given, computer pairwise distance to itself (B = A)
    """

    if B is None: B = A

    max_size = 2 ** 32
    n = A.size(0)
    m = B.size(0)
    d = A.size(1)

    if n * m * d <= max_size or m == 1:

        A_norm = torch.div(A.transpose(0, 1), A.norm(dim=1)).transpose(0, 1)
        B_norm = torch.div(B.transpose(0, 1), B.norm(dim=1)).transpose(0, 1)
        return torch.mm(A_norm, B_norm.transpose(0, 1))

    else:

        batch_size = max(1, max_size // (n * d))
        batch_results = []
        for i in range((m - 1) // batch_size + 1):
            id_left = i * batch_size
            id_rigth = min((i + 1) * batch_size, m)
            batch_results.append(outer_cosine_similarity(A, B[id_left:id_rigth]))

        return torch.cat(batch_results, dim=1)

# 1. считаем все попарные растояния
# Далее для каждого элемента
# 2. ищем близжайшие K+1 элементов
# 3. самый близжайший не расматриваем (это он сам)
# 4. смотрим метки для K близжайших кроме первого и проверяем совпадет ли эта метка с меткой того которого мы сейчас рассматриваем (это как раз 116 строка)
# 5. считаем долю таких совпадений
def metric_Recall_top_K(X, y, K, metric='cosine'):
    """
        calculate metric R@K
        X - tensor with size n x d, where n - number of examples, d - size of embedding vectors
        y - true labels (are the same for the same object with augmentations)
        N - count of closest examples, which we consider for recall calcualtion
        metric: 'cosine' / 'euclidean'.
            !!! 'euclidean' - to slow for datasets bigger than 100K rows
    """
    res_recall = []
    res_recall10 = []

    n = X.size(0)
    d = X.size(1)
    max_size = 2 ** 32
    batch_size = max(1, max_size // (n*d))

    with torch.no_grad():

        for i in range(1 + (len(X) - 1) // batch_size):

            id_left = i*batch_size
            id_right = min((i+1)*batch_size, len(y))
            y_batch = y[id_left:id_right]

            if metric == 'cosine':
                pdist = -1 * outer_cosine_similarity(X, X[id_left:id_right])
            elif metric == 'euclidean':
                pdist = outer_pairwise_distance(X, X[id_left:id_right])
            else:
                raise AttributeError(f'wrong metric "{metric}"')

            y_rep = y_batch.repeat(K, 1)

            values, indices = pdist.topk(K + 1, 0, largest=False)
            res_recall.append((y[indices[1:]] == y_rep).sum().item())

            values, indices = pdist.topk(int(n/10) + K + 1, 0, largest=False)
            y_rep = y_batch.repeat(int(n/10)+K, 1)
            res_recall10.append((y[indices[1:]] == y_rep).sum().item())

    recall = np.sum(res_recall) / len(y) / K
    recall10 = np.sum(res_recall10) / len(y) / K
    return recall, recall10


class NT_Xent(torch.nn.Module):
    def __init__(self, batch_size, temperature, device, world_size=1):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N в€’ 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        # if self.world_size > 1:
        #     z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

