import argparse
import os
import os.path as osp
import random

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CLASSES = (
    'background',
    'hazelnut',
    'forest',
    'permanent_cropland',
    'greenhouse',
    'grassland',
    'sparsely_vegetated_areas',
    'arable_land',
    'discontinuous_urban_fabric',
    'road_and_rail_networks',
    'water_courses',
    'water_bodies',
    'wetland',
)

PALETTE = {
    0: (0, 0, 0),
    1: (238, 128, 5),
    2: (128, 255, 0),
    3: (242, 166, 77),
    4: (204, 77, 242),
    5: (204, 242, 77),
    6: (204, 255, 204),
    7: (255, 255, 168),
    8: (255, 0, 0),
    9: (204, 0, 0),
    10: (0, 204, 242),
    11: (128, 242, 230),
    12: (166, 166, 255),
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HazelnutSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_suffix='.tif', mask_suffix='.tif', transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.items = self._match_items()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_id, mask_id = self.items[index]
        image_path = osp.join(self.image_dir, image_id + self.image_suffix)
        mask_path = osp.join(self.mask_dir, mask_id + self.mask_suffix)

        image = Image.open(image_path).convert('RGB')
        mask = self._load_mask(mask_path)

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            image = np.array(image)
            mask = np.array(mask)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()
        return image, mask

    def _match_items(self):
        image_ids = {
            self._normalize_id(filename): osp.splitext(filename)[0]
            for filename in os.listdir(self.image_dir)
            if filename.lower().endswith(self.image_suffix)
        }
        mask_ids = {
            self._normalize_id(filename): osp.splitext(filename)[0]
            for filename in os.listdir(self.mask_dir)
            if filename.lower().endswith(self.mask_suffix)
        }

        missing_masks = sorted(set(image_ids.keys()) - set(mask_ids.keys()))
        missing_images = sorted(set(mask_ids.keys()) - set(image_ids.keys()))
        if missing_masks:
            raise FileNotFoundError('Missing masks for images: {}'.format(missing_masks[:10]))
        if missing_images:
            raise FileNotFoundError('Missing images for masks: {}'.format(missing_images[:10]))

        return [(image_ids[key], mask_ids[key]) for key in sorted(image_ids.keys())]

    @staticmethod
    def _normalize_id(filename):
        name = osp.splitext(filename)[0]
        name = name.replace('_Image_', '_')
        name = name.replace('_mask_3band_', '_')
        return name

    @staticmethod
    def _load_mask(mask_path):
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 2:
            return Image.fromarray(mask.astype(np.uint8))

        mask = mask[:, :, :3]
        class_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        known_pixels = np.zeros(mask.shape[:2], dtype=bool)
        for class_id, color in PALETTE.items():
            pixels = np.all(mask == color, axis=-1)
            class_mask[pixels] = class_id
            known_pixels |= pixels

        if not np.all(known_pixels):
            unknown_count = np.size(known_pixels) - int(known_pixels.sum())
            raise ValueError('Mask has {} pixels with colors outside PALETTE: {}'.format(unknown_count, mask_path))
        return Image.fromarray(class_mask)


def get_train_transform(size):
    def transform(image, mask):
        image = image.resize((size, size), Image.BILINEAR)
        mask = mask.resize((size, size), Image.NEAREST)

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        rotations = random.randint(0, 3)
        if rotations:
            image = image.rotate(90 * rotations, expand=True)
            mask = mask.rotate(90 * rotations, expand=True)

        return np.array(image), np.array(mask)

    return transform


def get_val_transform(size):
    def transform(image, mask):
        image = image.resize((size, size), Image.BILINEAR)
        mask = mask.resize((size, size), Image.NEAREST)
        return np.array(image), np.array(mask)

    return transform


def confusion_matrix(pred, target, num_classes):
    mask = (target >= 0) & (target < num_classes)
    hist = torch.bincount(
        num_classes * target[mask].to(torch.int64) + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return hist


def metrics_from_hist(hist):
    hist = hist.float()
    intersection = torch.diag(hist)
    union = hist.sum(1) + hist.sum(0) - intersection
    iou = intersection / union.clamp_min(1)
    miou = iou.mean().item()
    oa = intersection.sum() / hist.sum().clamp_min(1)
    return miou, oa.item(), iou.cpu().numpy()


def train_one_epoch(model, loader, optimizer, loss_fn, device, num_classes):
    model.train()
    total_loss = 0.0
    hist = torch.zeros(num_classes, num_classes, device=device)

    for images, masks in tqdm(loader, desc='train', leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        hist += confusion_matrix(preds, masks, num_classes)

    miou, oa, _ = metrics_from_hist(hist)
    return total_loss / len(loader.dataset), miou, oa


@torch.no_grad()
def validate(model, loader, loss_fn, device, num_classes):
    model.eval()
    total_loss = 0.0
    hist = torch.zeros(num_classes, num_classes, device=device)

    for images, masks in tqdm(loader, desc='val', leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, masks)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        hist += confusion_matrix(preds, masks, num_classes)

    miou, oa, per_class_iou = metrics_from_hist(hist)
    return total_loss / len(loader.dataset), miou, oa, per_class_iou


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-images', default=r'C:\Users\U-III\Desktop\HazelNut\splitted2\train\images')
    parser.add_argument('--train-masks', default=r'C:\Users\U-III\Desktop\HazelNut\splitted2\train\masks')
    parser.add_argument('--val-images', default=r'C:\Users\U-III\Desktop\HazelNut\splitted2\val\images')
    parser.add_argument('--val-masks', default=r'C:\Users\U-III\Desktop\HazelNut\splitted2\val\masks')
    parser.add_argument('--test-images', default=r'C:\Users\U-III\Desktop\HazelNut\splitted2\test\images')
    parser.add_argument('--test-masks', default=r'C:\Users\U-III\Desktop\HazelNut\splitted2\test\masks')
    parser.add_argument('--output-dir', default='model_weights/smp_deeplabv3plus')
    parser.add_argument('--encoder', default='resnet34')
    parser.add_argument('--encoder-weights', default='imagenet')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    if device.type == 'cuda':
        print('gpu:', torch.cuda.get_device_name(0))

    train_dataset = HazelnutSegDataset(
        args.train_images,
        args.train_masks,
        transform=get_train_transform(args.image_size),
    )
    val_dataset = HazelnutSegDataset(
        args.val_images,
        args.val_masks,
        transform=get_val_transform(args.image_size),
    )

    print('train samples:', len(train_dataset))
    print('val samples:', len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )

    model = smp.DeepLabV3Plus(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=len(CLASSES),
    ).to(device)

    ce_loss = torch.nn.CrossEntropyLoss()
    dice_loss = smp.losses.DiceLoss(mode='multiclass', from_logits=True)

    def loss_fn(logits, masks):
        return ce_loss(logits, masks) + dice_loss(logits, masks)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_miou = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_miou, train_oa = train_one_epoch(model, train_loader, optimizer, loss_fn, device, len(CLASSES))
        val_loss, val_miou, val_oa, per_class_iou = validate(model, val_loader, loss_fn, device, len(CLASSES))
        scheduler.step()

        print(
            'epoch {}/{} | train_loss {:.4f} train_mIoU {:.4f} train_OA {:.4f} | '
            'val_loss {:.4f} val_mIoU {:.4f} val_OA {:.4f}'.format(
                epoch,
                args.epochs,
                train_loss,
                train_miou,
                train_oa,
                val_loss,
                val_miou,
                val_oa,
            )
        )
        print({name: float(iou) for name, iou in zip(CLASSES, per_class_iou)})

        checkpoint = {
            'model': model.state_dict(),
            'epoch': epoch,
            'classes': CLASSES,
            'palette': PALETTE,
            'encoder': args.encoder,
            'encoder_weights': args.encoder_weights,
            'image_size': args.image_size,
            'val_miou': val_miou,
            'val_oa': val_oa,
        }
        torch.save(checkpoint, osp.join(args.output_dir, 'last.pth'))
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(checkpoint, osp.join(args.output_dir, 'best.pth'))
            print('saved best.pth')


if __name__ == '__main__':
    main()
