import torch
import torch.nn.functional as F
from tqdm import tqdm

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    conf_matrix = torch.zeros((net.n_classes, net.n_classes), dtype=torch.int64, device=device)
    total_pixels = 0
    total_correct_pixels = 0
    total_correct_class_pixels = torch.zeros(net.n_classes, dtype=torch.int64, device=device)
    total_class_pixels = torch.zeros(net.n_classes, dtype=torch.int64, device=device)

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            total_pixels += mask_true.numel()
            total_correct_pixels += (mask_true == mask_true.argmax(dim=1)).sum().item()

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            mask_pred = net(image)
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

            conf_matrix += torch.einsum('bcwh,dcwh->bd', mask_pred, mask_true)
            total_correct_class_pixels += torch.einsum('bcwh->c', mask_pred * mask_true)
            total_class_pixels += torch.einsum('bcwh->c', mask_true)

    iou_scores = []
    pa = 0
    cpa_scores = []
    accuracy = 0
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for i in range(net.n_classes):
        true_positives = conf_matrix[i, i]
        false_positives = conf_matrix[i, :i].sum() + conf_matrix[i, i+1:].sum()
        false_negatives = conf_matrix[:i, i].sum() + conf_matrix[i+1:, i].sum()
        total = true_positives + false_positives + false_negatives
        if total > 0:
            iou = true_positives / total
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            iou = float('nan')
            precision = float('nan')
            recall = float('nan')
            f1_score = float('nan')

        iou_scores.append(iou)
        cpa_scores.append(total_correct_class_pixels[i] / total_class_pixels[i] if total_class_pixels[i] > 0 else 0)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1_score)

    pa = total_correct_pixels / total_pixels
    mpa = sum(cpa_scores) / net.n_classes
    miou = sum([iou for iou in iou_scores if not torch.isnan(iou)]) / len([iou for iou in iou_scores if not torch.isnan(iou)])
    accuracy = conf_matrix.trace() / conf_matrix.sum()
    precision = sum(precision_scores) / net.n_classes
    recall = sum(recall_scores) / net.n_classes
    f1_score = sum(f1_scores) / net.n_classes

    net.train()

    return {
        'PA': pa,
        'CPA': cpa_scores,
        'MPA': mpa,
        'IoU': iou_scores,
        'MIoU': miou,
        'Accuracy': accuracy,
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1-Score': f1_score
    }
