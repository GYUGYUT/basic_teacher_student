import torch

def calc_accuracy(X,Y):
    _, max_indices2 = torch.max(Y, 1)
    _, max_indices = torch.max(X, 1)
    train_acc = torch.sum(max_indices == max_indices2)/len(max_indices2)
    return train_acc