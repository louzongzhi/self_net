import torch
from tqdm import tqdm


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    iou_scores = [0] * net.n_classes
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_pred = net(image)
            mask_pred = mask_pred.argmax(dim=1)
            for i in range(1, net.n_classes):
                true_positives = torch.sum((mask_pred == i) & (mask_true == i))
                false_positives = torch.sum((mask_pred == i) & (mask_true != i))
                false_negatives = torch.sum((mask_pred != i) & (mask_true == i))
                iou_class = true_positives / (true_positives + false_positives + false_negatives + 1e-6)
                iou_scores[i] += iou_class.item()
    iou_scores = [score / num_val_batches for score in iou_scores]
    miou = sum(iou_scores[1:]) / (net.n_classes - 1)
    net.train()
    return iou_scores[1], iou_scores[2], iou_scores[3], miou