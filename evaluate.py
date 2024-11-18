import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_score import multiclass_dice_coeff, dice_coeff

def calculate_iou(mask_pred, mask_true, n_classes):
    ious = []
    for i in range(1, n_classes):  # Start from 1 to skip background
        intersection = ((mask_pred == i) & (mask_true == i)).sum().float()
        union = ((mask_pred == i) | (mask_true == i)).sum().float()
        iou = intersection / union if union > 0 else 0
        ious.append(iou)
    return ious

def calculate_accuracy(mask_pred, mask_true, n_classes):
    correct = (mask_pred == mask_true).sum().float()
    total = mask_true.numel() - (mask_true == 0).sum().float()  # Exclude background
    return correct / total if total > 0 else 0

def calculate_pixel_accuracy(mask_pred, mask_true, n_classes):
    correct_pixels_per_class = torch.zeros(n_classes - 1).to(mask_pred.device)  # Exclude background
    total_pixels_per_class = torch.zeros(n_classes - 1).to(mask_pred.device)
    for i in range(1, n_classes):  # Start from 1 to skip background
        correct_pixels_per_class[i - 1] = ((mask_pred == i) & (mask_true == i)).sum().float()
        total_pixels_per_class[i - 1] = (mask_true == i).sum().float()
    return correct_pixels_per_class / total_pixels_per_class

def calculate_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    accuracy_scores = []
    iou_scores = []
    pixel_accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes > 1:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                mask_true_one_hot = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

                # Calculate metrics for each class except background
                ious = calculate_iou(mask_pred.argmax(dim=1), mask_true, net.n_classes)
                iou_scores.extend(ious)
                pixel_accuracies.append(calculate_pixel_accuracy(mask_pred.argmax(dim=1), mask_true, net.n_classes))
                accuracy_scores.append(calculate_accuracy(mask_pred.argmax(dim=1), mask_true, net.n_classes))

                # Calculate precision and recall for F1 score
                for i in range(1, net.n_classes):
                    true_positives = ((mask_pred[:, i] == 1) & (mask_true_one_hot[:, i] == 1)).sum().float()
                    false_positives = ((mask_pred[:, i] == 1) & (mask_true_one_hot[:, i] == 0)).sum().float()
                    false_negatives = ((mask_pred[:, i] == 0) & (mask_true_one_hot[:, i] == 1)).sum().float()
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                    f1_scores.append(calculate_f1_score(precision, recall))

    # Calculate the average metrics
    dice_score /= max(num_val_batches, 1)

    # Calculate the average metrics
    dice_score /= max(num_val_batches, 1)
    accuracy_scores = torch.tensor(accuracy_scores).mean().item()
    iou_scores = torch.tensor(iou_scores).mean().item()
    pixel_accuracies = torch.stack(pixel_accuracies).mean(dim=0)
    f1_scores = torch.tensor(f1_scores).mean().item()

    net.train()
    return dice_score, accuracy_scores, iou_scores, pixel_accuracies.tolist(), f1_scores
