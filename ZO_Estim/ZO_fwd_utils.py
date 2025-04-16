import torch

trainable_layers_dict = {
    'nn.Linear': torch.nn.Linear,
    'nn.Conv2d': torch.nn.Conv2d,
}

def get_iterable_block_name():
    return 'net'

def ZO_pre_block_forward(model, x):
    x = torch.flatten(x,1)
    return x

def ZO_post_block_forward(model, x):
    x = torch.flatten(x,1)
    return x