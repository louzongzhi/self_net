import torch
import torch.nn.functional as F
from tqdm import tqdm

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    conf_matrix = torch.zeros((net.n_classes, net.n_classes), dtype=torch.int64, device=device)

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            mask_pred = net(image)
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), num_classes=net.n_classes).permute(0, 3, 1, 2).float()
            mask_true = F.one_hot(mask_true, num_classes=net.n_classes).permute(0, 3, 1, 2).float()

            conf_matrix += torch.einsum('bcwh,dcwh->bd', mask_pred, mask_true)

    iou_scores = []

    for i in range(net.n_classes):
        true_positives = conf_matrix[i, i]
        false_positives = conf_matrix[i, :i].sum() + conf_matrix[i, i+1:].sum()
        false_negatives = conf_matrix[:i, i].sum() + conf_matrix[i+1:, i].sum()
        total = true_positives + false_positives + false_negatives
        if total > 0:
            iou = true_positives / total
        else:
            iou = float('nan')

        iou_scores.append(iou)
    miou = sum([iou for iou in iou_scores[1:] if not torch.isnan(iou)]) / len([iou for iou in iou_scores[1:] if not torch.isnan(iou)])

    net.train()

    return {
        'IoU': iou_scores,
        'MIoU': miou,
    }
