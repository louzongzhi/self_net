import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import warnings
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluate import evaluate
from models import load_model
from data_loading import BasicDataset, CarvanaDataset
from dice_score import dice_loss

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)


dir_img = Path('./NEU_Seg/images')
dir_img_train = os.path.join(dir_img, 'training')
dir_img_val = os.path.join(dir_img, 'test')
dir_mask = Path('./NEU_Seg/annotations')
dir_mask_train = os.path.join(dir_mask, 'training')
dir_mask_val = os.path.join(dir_mask, 'test')
dir_checkpoint = Path('./checkpoints/')
if not os.path.exists(dir_checkpoint):
    os.makedirs(dir_checkpoint)
dir_checkpoint_history = os.path.join(dir_checkpoint, 'history')
if not os.path.exists(dir_checkpoint_history):
    os.mkdir(dir_checkpoint_history)
dir_checkpoint_best = os.path.join(dir_checkpoint, 'best')
if not os.path.exists(dir_checkpoint_best):
    os.mkdir(dir_checkpoint_best)


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        gradient_clipping: float = 1.0,
):
    try:
        dataset_train = CarvanaDataset(dir_img_train, dir_mask_train, img_scale)
        dataset_val = CarvanaDataset(dir_img_val, dir_mask_val, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset_train = BasicDataset(dir_img_train, dir_mask_train, img_scale)
        dataset_val = BasicDataset(dir_img_val, dir_mask_val, img_scale)

    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(dataset_train, shuffle=True, **loader_args)
    val_loader = DataLoader(dataset_val, shuffle=False, drop_last=True, **loader_args)

    experiment = wandb.init(
        project='钢材表面缺陷检测与分割',
        resume='allow',
        anonymous='must'
    )
    experiment.config.update(
        dict(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_percent=val_percent,
            save_checkpoint=save_checkpoint,
            img_scale=img_scale,
            amp=amp
        )
    )

    logging.info(
        f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        '''
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        # momentum=momentum,
        foreach=True
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    best_score = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(dataset_train), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                division_step = (len(dataset_train) // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        dice_score, accuracy_scores, iou_scores, pixel_accuracies, f1_scores = evaluate(model, val_loader, device, amp)
                        val_score = iou_scores
                        scheduler.step(val_score)
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': dice_score,
                                'accuracy': accuracy_scores,
                                'iou': iou_scores,
                                'pixel_accuracy': pixel_accuracies,
                                'f1': f1_scores,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })

                            file_path = os.path.join(dir_checkpoint, 'runs.csv')
                            if not os.path.exists(file_path):
                                with open(file_path, 'w') as f:
                                    f.write("epoch,dice_score,accuracy_scores,iou_scores,pixel_accuracies,f1_scores\n")
                            with open(file_path, 'a') as f:
                                f.write(f"{epoch},{dice_score},{accuracy_scores},{iou_scores},{pixel_accuracies},{f1_scores}\n")

                        except:
                            pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f'{dir_checkpoint_history}/checkpoint_{epoch}.pth')
            logging.info(f'Checkpoint {epoch} saved!')

        best_epoch = 0
        if val_score > best_score:
            best_epoch = epoch
            best_score = val_score
            torch.save(model.state_dict(), f'{dir_checkpoint_best}/model.pth')
            logging.info(f'\nmiou:\t{val_score}\nBest\tcheckpoint\t{epoch}\tsaved!\n')
        if epoch - best_epoch > 10:
            break


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4, help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--classes', '-c', type=int, default=4, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logging.info(f'Using GPU {torch.cuda.get_device_name(0)}')
        else:
            import torch_directml
            device = torch_directml.device()
            logging.info(f'Using DirectML')
    except Exception as e:
        logging.info(f'Could not use GPU with error {e}')
        device = torch.device('cpu')
    logging.info(f'Using device {device}')

    model = load_model(model_name='UNet', n_channels=3, num_classes=args.classes)
    model = model.to(device=device)
    model = model.to(memory_format=torch.channels_last)

    logging.info(
        f'Network:\n'
        f'\t{model.n_channels} input channels\n'
        f'\t{model.n_classes} output channels (classes)\n'
    )

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error(
            'Detected OutOfMemoryError! '
            'Enabling checkpointing to reduce memory usage, but this slows down training. '
            'Consider enabling AMP (--amp) for fast and memory efficient training'
        )
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
