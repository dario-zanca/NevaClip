import torch

def cosine_sim(x, y):
    val = torch.nn.functional.cosine_similarity(x, y, 1)
    return -val + 1

def target_function(x, y):
    return y
