import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    images, captions = zip(*batch)
    
    # Pad captions to the same length
    padded_captions = pad_sequence(captions, batch_first=True, padding_value=0)
    
    # Stack the images into a tensor
    images = torch.stack(images, dim=0)
    
    return images, padded_captions
