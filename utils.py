import torch

def image_int_to_float(x):
    return x / 127.5 - 1.

def image_float_to_int(x):
    return torch.round( (x+1) * 127.5 ).long()

def text_int_to_str(x, with_mask=False):
    mp = [chr(ord('a') + i) for i in range(26)]
    if with_mask:
        mp = ['_', ' '] + mp
    else:
        mp = [' '] + mp

    return [''.join([mp[t] for t in row]) for row in x]