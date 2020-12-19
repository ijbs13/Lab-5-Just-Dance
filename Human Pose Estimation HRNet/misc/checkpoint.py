import os
import torch


def save_checkpoint(path, epoch, model, optimizer, params=None):
    
    if os.path.isdir(path):
        path = os.path.join(path, 'checkpoint.pth')
    torch.save(
        {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'params': params,
        },
        path
    )


def load_checkpoint(path, model, optimizer=None, device=None):
    
    if os.path.isdir(path):
        path = os.path.join(path, 'checkpoint.pth')
    checkpoint = torch.load(path, map_location=device)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    params = checkpoint['params']

    return epoch, model, optimizer, params
