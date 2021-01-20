import numpy as np
import torch
from load_data import load_data
from util import kernel
import matplotlib.pyplot as plt
from evaluate import func


def partial_g(m, g, d, X, K_P, H, alpha, beta):
    G = torch.diag(g) @ (torch.eye(d) - torch.diag(m)) @ X
    K_G = kernel(G)

    f = H @ K_P @ H @ K_G
    f = -alpha * torch.trace(f)
    f.backward()
    diff = g.grad

    if diff == None:
        print('Error. No gradient.')
        return

    diff += beta
    return diff


def partial_m(m, g, d, X, K_X, K_P, H, alpha, beta):
    G = torch.diag(g) @ (torch.eye(d) - torch.diag(m)) @ X
    M = torch.diag(m) @ X
    K_G = kernel(G)
    K_M = kernel(M)

    f = -1 * torch.trace(H @ K_X @ H @ K_M) + alpha * torch.trace(H @ K_P @ H @ K_M) - alpha * torch.trace(
        H @ K_P @ H @ K_G)
    f.backward()
    diff = m.grad

    if diff == None:
        print('Error. No gradient.')
        return

    diff += beta
    return diff


def lr_scheduler(epoch):
    if epoch <= 100:
        t = 1e-2
    elif epoch <= 200:
        t = 1e-3
    else:
        t = 1e-4
    return t


def fufs(data, protect_attribute, **kwargs):
    if 'iters' not in kwargs:
        iters = 50
    else:
        iters = kwargs['iters']
    if 'alpha' not in kwargs:
        alpha = 1
    else:
        alpha = kwargs['alpha']

    if 'beta' not in kwargs:
        beta = 0.1
    else:
        beta = kwargs['beta']

    # seperate protected and non-protected data
    P = data[:, protect_attribute]
    X = np.delete(data, protect_attribute, 1)
    X = torch.tensor(X, dtype=float).T
    P = torch.tensor(P, dtype=float).T

    d = X.shape[0]  # number of features
    n = X.shape[1]  # number of data samples
    # print('d:', d)
    # print('n:', n)

    # centralized kernel matrix
    matrix_1 = torch.ones((n, n), dtype=float)
    I = torch.eye(n, dtype=float)
    H = I - matrix_1 / n

    # calculate kernel matrix of non-protected input data
    K_X = kernel(X)
    # kernel matrix of protected data
    K_P = kernel(P)

    # initialize
    m = torch.ones(d, requires_grad=True, dtype=float)
    g = torch.ones(d, requires_grad=True, dtype=float)

    for epoch in range(iters):
        t = lr_scheduler(epoch)

        grad_g = partial_g(m, g, d, X, K_P, H, alpha, beta)
        g = g - t * grad_g
        g[g > 1] = 1
        g[g < 0] = 0
        g = torch.tensor(g, requires_grad=True)
        # g = g.clone().detach().requires_grad_(True)

        grad_m = partial_m(m, g, d, X, K_X, K_P, H, alpha, beta)
        m = m - t * grad_m
        m[m > 1] = 1
        m[m < 0] = 0
        m = torch.tensor(m, requires_grad=True)
        # m = m.clone().detach().requires_grad_(True)

    return m, g


if __name__ == '__main__':
    data, protect_attribute, true_label, n_cluster, protect_value = load_data('toxic')

    m, g = fufs(data, protect_attribute, alpha=0.01, beta=10)
    m = m.detach().numpy()
    idx = np.argsort(-m, 0)
    acc, nmi, bal, prop = func(idx, k, data, protect_attribute, true_label, n_cluster, protect_value)
