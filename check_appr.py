import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sklearn
from sklearn import datasets
from sklearn.utils import shuffle
import teneva
from time import perf_counter as tpc
import torch
import tqdm


from ht.node import Node


def calc_ht(n, r, m_trn, I_trn, y_trn, I_tst, y_tst, epochs, batch, lr, device):
    _time = tpc()

    I_trn = torch.tensor(I_trn, dtype=torch.long, device=device)
    y_trn = torch.tensor(y_trn, dtype=torch.float, device=device)
    I_tst = torch.tensor(I_tst, dtype=torch.long, device=device)
    y_tst = torch.tensor(y_tst, dtype=torch.float, device=device)

    Y = Node.random(n, r, device=device)
    opt = torch.optim.Adam(Y.to_core_list(grad=True), lr=lr)

    def loss_func(I, y):
        y_pred = Y.get(I)
        return torch.sum((y_pred - y)**2)

    for ep in range(epochs):
        perm = torch.randperm(I_trn.shape[0])
        I_trn_cur = I_trn[perm]
        y_trn_cur = y_trn[perm]

        for j in range(len(I_trn_cur) // batch):
            loss = loss_func(
                I_trn_cur[j * batch: (j+1)*batch],
                y_trn_cur[j * batch: (j+1)*batch])
            loss.backward()

            l = loss.detach().numpy()
            #print(l)

            opt.step()
            opt.zero_grad()

    with torch.no_grad():
        y_our = Y.get(I_tst)
        e = (torch.norm(y_our - y_tst) / torch.norm(y_tst)).item()
    return e, tpc() - _time


def calc_tt(f, n, m_trn, I_tst, y_tst):
    _time = tpc()

    Y0 = teneva.tensor_rand(n, r=1)
    Y = teneva.cross(f, Y0, m_trn)

    y_own = teneva.get_many(Y, I_tst)
    e = np.linalg.norm(y_own - y_tst) / np.linalg.norm(y_tst)

    return e, tpc() - _time


def check(d=16, n=10, r=[2, 2, 2, 2], m_trn=10000, m_tst=1000, epochs=100, batch=100, lr=1.E-2, device='cpu'):
    for func in teneva.func_demo_all(d):
        func.set_grid(n, kind='uni')
        func.build_trn_ind(m_trn)
        func.build_tst_ind(m_tst)

        f = func.get_f_ind
        I_trn, y_trn = func.I_trn_ind, func.Y_trn_ind
        I_tst, y_tst = func.I_tst_ind, func.Y_tst_ind

        e_tt, t_tt = calc_tt(f, [n]*d, m_trn, I_tst, y_tst)
        e_ht, t_ht = calc_ht([n]*d, r, m_trn, I_trn, y_trn, I_tst, y_tst,
            epochs, batch, lr, device=device)

        name = f'{d:-3}D-' + func.name
        text = '  - ' + name + ' ' * max(0, 22-len(name)-4) + ' | '
        text += f'TT e = {e_tt:-7.1e}, t = {t_tt:-7.2f} | '
        text += f'HT e = {e_ht:-7.1e}, t = {t_ht:-7.2f}'
        print(text)
        #break
    print('\n\n')


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
