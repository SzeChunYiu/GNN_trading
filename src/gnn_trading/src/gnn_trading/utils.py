import numpy as np
import random
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pick_device(pref='auto'):
    if pref == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return pref
