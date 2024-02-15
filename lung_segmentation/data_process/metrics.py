from sklearn.metrics import f1_score, recall_score, precision_score
import numpy as np

def calculate_metrics(pred, target):
    
    pred_np = pred.cpu().detach().numpy()
    target_np = target.cpu().detach().numpy()

    pred_binary = (pred_np > 0.5).astype(np.uint8)
    target_binary = (target_np > 0.5).astype(np.uint8)

    f1 = f1_score(target_binary.flatten(), pred_binary.flatten())
    recall = recall_score(target_binary.flatten(), pred_binary.flatten())
    precision = precision_score(target_binary.flatten(), pred_binary.flatten())
    
    # Dice coefficient
    intersection = np.logical_and(target_binary, pred_binary).sum()
    dice = (2.0 * intersection) / (target_binary.sum() + pred_binary.sum())

    # IoU (Jaccard Index)
    union = np.logical_or(target_binary, pred_binary).sum()
    iou = intersection / union

    return f1, recall, precision, dice, iou

def calculate_accuracy(outputs, targets, threshold=0.5):
    predictions = (outputs > threshold).float()
    correct = (predictions == targets).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy.item()