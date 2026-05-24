import argparse
import csv
import os
import os.path as osp

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CLASSES = (
    'background', 'hazelnut', 'forest', 'permanent_cropland', 'greenhouse',
    'grassland', 'sparsely_vegetated_areas', 'arable_land',
    'discontinuous_urban_fabric', 'road_and_rail_networks', 'water_courses',
    'water_bodies', 'wetland',
)

PALETTE = {
    0: (0, 0, 0), 1: (238, 128, 5), 2: (128, 255, 0),
    3: (242, 166, 77), 4: (204, 77, 242), 5: (204, 242, 77),
    6: (204, 255, 204), 7: (255, 255, 168), 8: (255, 0, 0),
    9: (204, 0, 0), 10: (0, 204, 242), 11: (128, 242, 230),
    12: (166, 166, 255),
}


class HazelnutTestDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=512, image_suffix='.tif', mask_suffix='.tif'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix
        self.items = self._match_items()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_id, mask_id = self.items[index]
        image_path = osp.join(self.image_dir, image_id + self.image_suffix)
        mask_path = osp.join(self.mask_dir, mask_id + self.mask_suffix)
        image_original = np.array(Image.open(image_path).convert('RGB'))
        mask_original = np.array(self._load_mask(mask_path))
        image = Image.fromarray(image_original).resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = Image.fromarray(mask_original).resize((self.image_size, self.image_size), Image.NEAREST)
        width, height = Image.fromarray(image_original).size
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask, image_id, width, height, image_original, mask_original

    def _match_items(self):
        image_ids = {self._normalize_id(f): osp.splitext(f)[0] for f in os.listdir(self.image_dir) if f.lower().endswith(self.image_suffix)}
        mask_ids = {self._normalize_id(f): osp.splitext(f)[0] for f in os.listdir(self.mask_dir) if f.lower().endswith(self.mask_suffix)}
        missing_masks = sorted(set(image_ids) - set(mask_ids))
        missing_images = sorted(set(mask_ids) - set(image_ids))
        if missing_masks:
            raise FileNotFoundError('Missing masks for images: {}'.format(missing_masks[:10]))
        if missing_images:
            raise FileNotFoundError('Missing images for masks: {}'.format(missing_images[:10]))
        return [(image_ids[k], mask_ids[k]) for k in sorted(image_ids)]

    @staticmethod
    def _normalize_id(filename):
        name = osp.splitext(filename)[0]
        return name.replace('_Image_', '_').replace('_mask_3band_', '_')

    @staticmethod
    def _load_mask(mask_path):
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 2:
            return Image.fromarray(mask.astype(np.uint8))
        mask = mask[:, :, :3]
        class_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        known = np.zeros(mask.shape[:2], dtype=bool)
        for class_id, color in PALETTE.items():
            pixels = np.all(mask == color, axis=-1)
            class_mask[pixels] = class_id
            known |= pixels
        if not np.all(known):
            raise ValueError('Mask has colors outside PALETTE: {}'.format(mask_path))
        return Image.fromarray(class_mask)


def confusion_matrix(pred, target, num_classes):
    mask = (target >= 0) & (target < num_classes)
    return torch.bincount(num_classes * target[mask].to(torch.int64) + pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)


def metrics_from_hist(hist):
    hist = hist.float()
    intersection = torch.diag(hist)
    union = hist.sum(1) + hist.sum(0) - intersection
    iou = intersection / union.clamp_min(1)
    precision = intersection / hist.sum(0).clamp_min(1)
    recall = intersection / hist.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-7)
    oa = intersection.sum() / hist.sum().clamp_min(1)
    return {'mIoU': iou.mean().item(), 'mF1': f1.mean().item(), 'OA': oa.item(), 'IoU': iou.cpu().numpy(), 'F1': f1.cpu().numpy()}


def mask_to_rgb(mask):
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in PALETTE.items():
        rgb[mask == class_id] = color
    return rgb


def save_triplet(image, mask_true, mask_pred, output_path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, title, visual in zip(axes, ['Image', 'Ground Truth', 'Model Prediction'], [image, mask_to_rgb(mask_true), mask_to_rgb(mask_pred)]):
        ax.imshow(visual)
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_confusion_matrix(hist, output_dir):
    import matplotlib.pyplot as plt
    hist_np = hist.cpu().numpy().astype(np.int64)
    normalized = hist_np.astype(np.float64) / np.maximum(hist_np.sum(axis=1, keepdims=True), 1)
    for path, matrix in [(osp.join(output_dir, 'confusion_matrix_counts.csv'), hist_np), (osp.join(output_dir, 'confusion_matrix_normalized.csv'), normalized)]:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['true/pred'] + list(CLASSES))
            for class_name, row in zip(CLASSES, matrix):
                writer.writerow([class_name] + list(row))
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(normalized, interpolation='nearest', cmap='Blues', vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Normalized Confusion Matrix - UNet++')
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(CLASSES, fontsize=8)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            value = normalized[i, j]
            if value >= 0.01 or i == j:
                ax.text(j, i, '{:.2f}'.format(value), ha='center', va='center', color='white' if value > 0.5 else 'black', fontsize=7)
    fig.tight_layout()
    fig.savefig(osp.join(output_dir, 'confusion_matrix_normalized.png'), dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-images', default=r'C:\Users\U-III\Desktop\HazelNut\splitted2\test\images')
    parser.add_argument('--test-masks', default=r'C:\Users\U-III\Desktop\HazelNut\splitted2\test\masks')
    parser.add_argument('--checkpoint', default=r'C:\Users\U-III\Desktop\geoseg\unetplusplus\best.pth')
    parser.add_argument('--output-dir', default=r'C:\Users\U-III\Desktop\geoseg\unetplusplus\test_results')
    parser.add_argument('--encoder', default=None)
    parser.add_argument('--image-size', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--save-predictions', action='store_true')
    parser.add_argument('--save-triplets', action='store_true')
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    prediction_dir = osp.join(args.output_dir, 'predictions')
    triplet_dir = osp.join(args.output_dir, 'triplets')
    if args.save_predictions:
        os.makedirs(prediction_dir, exist_ok=True)
    if args.save_triplets:
        os.makedirs(triplet_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    if device.type == 'cuda':
        print('gpu:', torch.cuda.get_device_name(0))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    encoder = args.encoder or checkpoint.get('encoder', 'resnet34')
    image_size = args.image_size or checkpoint.get('image_size', 512)
    model = smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=len(CLASSES)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    dataset = HazelnutTestDataset(args.test_images, args.test_masks, image_size=image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    print('test samples:', len(dataset))
    hist = torch.zeros(len(CLASSES), len(CLASSES), device=device)
    for images, masks, image_ids, widths, heights, original_images, original_masks in tqdm(loader, desc='test'):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        preds = model(images).argmax(dim=1)
        hist += confusion_matrix(preds, masks, len(CLASSES))
        preds_np = preds.cpu().numpy().astype(np.uint8)
        for pred, image_id, width, height, original_image, original_mask in zip(preds_np, image_ids, widths, heights, original_images, original_masks):
            pred_img = Image.fromarray(pred).resize((int(width), int(height)), Image.NEAREST)
            pred_np = np.array(pred_img)
            if args.save_predictions:
                pred_img.save(osp.join(prediction_dir, image_id + '_pred_ids.png'))
                Image.fromarray(mask_to_rgb(pred_np)).save(osp.join(prediction_dir, image_id + '_pred_rgb.png'))
            if args.save_triplets:
                save_triplet(np.array(original_image), np.array(original_mask), pred_np, osp.join(triplet_dir, image_id + '_triplet.png'))
    metrics = metrics_from_hist(hist)
    save_confusion_matrix(hist, args.output_dir)
    print('test mIoU:', metrics['mIoU'])
    print('test mF1:', metrics['mF1'])
    print('test OA:', metrics['OA'])
    lines = ['test mIoU: {:.6f}'.format(metrics['mIoU']), 'test mF1: {:.6f}'.format(metrics['mF1']), 'test OA: {:.6f}'.format(metrics['OA']), '', 'per-class metrics:']
    for name, iou, f1 in zip(CLASSES, metrics['IoU'], metrics['F1']):
        line = '{} | IoU: {:.6f} | F1: {:.6f}'.format(name, float(iou), float(f1))
        print(line)
        lines.append(line)
    with open(osp.join(args.output_dir, 'test_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
