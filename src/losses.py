import torch
import torch.nn as nn
import numpy as np

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0,3,1,2).float()
        dims = (0,2,3)
        inter = torch.sum(probs * onehot, dims)
        card = torch.sum(probs + onehot, dims)
        dice = (2*inter + self.smooth) / (card + self.smooth)
        return 1.0 - dice.mean()

def class_weights_from_mask(mask_np, num_classes, eps=1e-6):
    flat = mask_np.reshape(-1)
    counts = np.array([(flat == c).sum() for c in range(num_classes)], dtype=np.float64) + eps
    inv = 1.0 / counts
    w = inv / inv.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)
