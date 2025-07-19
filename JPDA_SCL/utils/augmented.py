import torch
import numpy as np


def radiation_noise(data, alpha_range=(0.9, 1.1), beta=0.04): 
    data_np = data.cpu().numpy()
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data_np.shape)
    x = alpha * data_np + beta * noise
    return torch.tensor(x, dtype=torch.float32).to(data.device)

def flip_augmentation(data): 
    data= data.cpu().numpy()
    horizontal = np.random.random() > 0.5 
    vertical = np.random.random() <= 0.5 
    if horizontal:
        data = np.fliplr(data)      
        data = torch.from_numpy(data.copy()) 
       
    if vertical:
        data = np.flipud(data)     
        data = torch.from_numpy(data.copy())     
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).cuda() 
    return data.cuda() 