import torch
import torch.nn.functional as F
from tqdm import tqdm
import time

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    iou_scores = [0.0] * net.n_classes
    total_time = 0.0

    start_time = time.time()

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            mask_pred = net(image)

            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                intersection = (mask_pred * mask_true).sum()
                union = mask_pred.sum() + mask_true.sum() - intersection
                iou_scores[1] += (intersection / union) if union > 0 else 0
            else:
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                for i in range(net.n_classes):
                    intersection = (mask_pred[:, i] * mask_true[:, i]).sum()
                    union = mask_pred[:, i].sum() + mask_true[:, i].sum() - intersection
                    iou_scores[i] += (intersection / union) if union > 0 else 0

    total_time = time.time() - start_time
    fps = num_val_batches / total_time if total_time > 0 else 0

    miou = sum(iou_scores[1:4]) / 3

    model_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)

    net.train()

    iou_class0, iou_class1, iou_class2, iou_class3 = iou_scores[0], iou_scores[1], iou_scores[2], iou_scores[3]

    return iou_class0, iou_class1, iou_class2, iou_class3, miou, fps, model_parameters
