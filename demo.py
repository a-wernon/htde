import matplotlib.pyplot as plt
import numpy as np
import os
import random
from time import perf_counter as tpc
import torch


from ht.node import Node


def demo(k=1.E+5, device='cpu'):
    # Create random HT-tensor:
    Y = Node.random(n=[2]*8, r=[2, 3, 4], device=device)
    print(f'\nGenerated random HT-tensor:')
    print(Y)

    # Convert it to the full format:
    Y_full = Y.full()
    print('\nShape of the full tensor: ', Y_full.shape)

    # Get batch of values for multi-indices:
    I = torch.tensor([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=torch.long, device=device)
    y = Y.get(I)
    print('\nValues of HT-tensor in several points: \n', y)

    # Check scalar product:
    v1 = torch.sum(Y_full * Y_full).item()
    v2 = Y.scalar_product().item()
    e = abs((v1-v2) / v1)
    print(f'\nScalar product error: {e:-7.1e}')

    # Sample from the tensor:
    _time = tpc()
    I = [Y.sample() for _ in range(int(k))]
    print(f'\nSample from HT-tensor done. Time : {tpc()-_time:-8.4f}')

    # Plot the result:
    pow2 = 2**np.arange(8)[::-1]
    counts = np.unique([i @ pow2 for i in I], return_counts=True)[1]
    plt.plot(counts/sum(counts), linewidth=2)

    y_full_ = Y_full.cpu().flatten()
    plt.plot(y_full_/sum(y_full_), '--', linewidth=1)
    plt.savefig('result/sample_node.png')


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

    demo(device=device)
