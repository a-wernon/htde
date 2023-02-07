import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sklearn
from sklearn import datasets
from sklearn.utils import shuffle
from time import perf_counter as tpc
import torch
import tqdm


from ht.node import Node


def build_data(m, n):
    X_trn = sklearn.datasets.make_moons(n_samples=m, shuffle=True,
        noise=0.003, random_state=None)[0]
    X_trn -= X_trn.mean(axis = 0)
    X_trn *= ((np.pi - 1) / np.abs(X_trn).max(axis = 0))

    a = np.min(X_trn)
    b = np.max(X_trn)
    I_trn = np.array([poi_to_ind(x, a, b, n) for x in X_trn])

    return I_trn


def check(m=10000, n=100, r=3, epochs=20, batch=100, lr=1.E-4, samples=1.E+3, device='cpu'):
    I_trn = build_data(m, n)

    Y = Node.random(n=[n]*2, r=[r], device=device)

    def loss_func(I):
        return Y.scalar_product() - 2 * torch.sum(Y.get(I))

    params = [G.requires_grad_() for G in Y.to_core_list()]
    optimizer = torch.optim.Adam(params, lr=lr)

    for _ in tqdm.tqdm(range(epochs)):
        I_trn_cur = shuffle(I_trn)
        for j in range(len(I_trn_cur) // batch):
            loss = loss_func(I_trn_cur[j * batch: (j+1)*batch])
            loss.backward()

            l = loss.detach().numpy()
            print(l)

            optimizer.step()
            optimizer.zero_grad()

    I_gen = []
    for _ in tqdm.tqdm(range(int(samples))):
        I_gen.append(Y.sample())
    I_gen = torch.tensor(I_gen)

    plt.scatter(I_trn[:, 0], I_trn[:, 1])
    plt.scatter(I_gen[:, 0], I_gen[:, 1])
    plt.savefig('result/check_opt_2d.png')


def poi_to_ind(X, a, b, n):
    d = X.shape[0]
    n = np.ones(d) * n

    X_sc = (X - a) / (b - a)
    X_sc[X_sc < 0.] = 0.
    X_sc[X_sc > 1.] = 1.

    I = X_sc * (n - 1)
    I = np.rint(I)
    I = np.array(I, dtype=int)

    I[I < 0] = 0
    I[I > n-1] = n[I > n-1] - 1

    return I


if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs('result', exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    check(device=device)
