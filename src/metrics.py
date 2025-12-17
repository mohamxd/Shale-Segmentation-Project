import math, numpy as np, torch
from sklearn.metrics import confusion_matrix

@torch.inference_mode()
def batch_metrics(logits, targets, num_classes):
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1).cpu().numpy().reshape(-1)
    gts = targets.cpu().numpy().reshape(-1)
    per_iou, per_dice = [], []
    for c in range(num_classes):
        tp = int(((preds == c) & (gts == c)).sum())
        fn = int(((preds != c) & (gts == c)).sum())
        fp = int(((preds == c) & (gts != c)).sum())
        union = tp + fp + fn
        iou = tp / union if union > 0 else float('nan')
        denom = (2*tp + fp + fn)
        dice = (2*tp) / denom if denom > 0 else float('nan')
        per_iou.append(iou); per_dice.append(dice)
    return float(np.nanmean(per_iou)), float(np.nanmean(per_dice))

def confusion(y_true, y_pred, num_classes):
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
