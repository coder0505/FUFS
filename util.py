import torch

#calculate kernel metrices
#default RBF kernel

def kernel(X):
    sigma = 1
    A = X.T @ X
    d = torch.diag(A)
    B = d[:,None] + d[None,:] -2 * A
    B = -B / sigma**2
    K = torch.exp(B)
    return K
