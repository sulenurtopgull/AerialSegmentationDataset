import argparse
import csv
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from geoseg.datasets.custom_dataset import CLASSES, PALETTE, CustomSegDataset, val_aug
from geoseg.models.UNetFormer import UNetFormer


def mask_to_rgb(mask):
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in PALETTE.items():
        rgb[mask == class_id] = color
    return rgb


def confusion_matrix(pred, target, num_classes):
    mask = (target >= 0) & (target < num_classes)
    return torch.bincount(
        num_classes * target[mask].to(torch.int64) + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)


def metrics_from_hist(hist):
    hist = hist.float()
    intersection = torch.diag(hist)
    union = hist.sum(1) + hist.sum(0) - intersection
    iou = intersection / union.clamp_min(1)
    precision = intersection / hist.sum(0).clamp_min(1)
    recall = intersection / hist.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-7)
    oa = intersection.sum() / hist.sum().clamp_min(1)
    return {
        'mIoU': iou.mean().item(),
        'mF1': f1.mean().item(),
        'OA': oa.item(),
        'IoU': iou.cpu().numpy(),
        'F1': f1.cpu().numpy(),
    }


def write_confusion_matrix(hist, output_dir):
    hist_np = hist.cpu().numpy().astype(np.int64)
    normalized = hist_np.astype(np.float64) / np.maximum(hist_np.sum(axis=1, keepdims=True), 1)
    for filename, matrix in [('confusion_matrix_counts.csv', hist_np), ('confusion_matrix_normalized.csv', normalized)]:
        with open(osp.join(output_dir, filename), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['true/pred'] + list(CLASSES))
            for class_name, row in zip(CLASSES, matrix):
                writer.writerow([class_name] + list(row))

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(normalized, interpolation='nearest', cmap='Blues', vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Normalized Confusion Matrix - UNetFormer')
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
                ax.text(j, i, '{:.2f}'.format(value), ha='center', va='center',
                        color='white' if value > 0.5 else 'black', fontsize=7)
    fig.tight_layout()
    fig.savefig(osp.join(output_dir, 'confusion_matrix_normalized.png'), dpi=200)
    plt.close(fig)


def save_triplet(image_path, mask_path, pred, output_path):
    image = Image.open(image_path).convert('RGB')
    mask = CustomSegDataset.load_mask(None, mask_path)
    mask = np.array(mask)
    mask_rgb = Image.fromarray(mask_to_rgb(mask))
    pred_rgb = Image.fromarray(mask_to_rgb(pred))

    panel_size = (420, 420)
    panels = []
    for panel in [image, mask_rgb, pred_rgb]:
        panel = panel.convert('RGB')
        panel.thumbnail(panel_size, Image.BILINEAR)
        canvas = Image.new('RGB', panel_size, 'white')
        canvas.paste(panel, ((panel_size[0] - panel.width) // 2, (panel_size[1] - panel.height) // 2))
        panels.append(canvas)

    title_h = 42
    output = Image.new('RGB', (panel_size[0] * 3, panel_size[1] + title_h), 'white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(output)
    labels = ['Image', 'Ground Truth', 'UNetFormer Prediction']
    for idx, (label, panel) in enumerate(zip(labels, panels)):
        x = idx * panel_size[0]
        output.paste(panel, (x, title_h))
        draw.text((x + 12, 12), label, fill=(0, 0, 0))
    output.save(output_path)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model = UNetFormer(num_classes=len(CLASSES), pretrained=False).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-images', default=r'C:\Users\CBS-7\Desktop\FINDIK_PROJE\AERIAL_DATA_CIKTILAR\splitted2\splitted2\test\images')
    parser.add_argument('--test-masks', default=r'C:\Users\CBS-7\Desktop\FINDIK_PROJE\AERIAL_DATA_CIKTILAR\splitted2\splitted2\test\masks')
    parser.add_argument('--checkpoint', default=r'model_weights\custom\unetformer-custom-512crop-e50\best.pth')
    parser.add_argument('--output-dir', default=r'unetformer\test_results')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--save-predictions', action='store_true')
    parser.add_argument('--save-triplets', action='store_true')
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pred_dir = osp.join(args.output_dir, 'predictions')
    triplet_dir = osp.join(args.output_dir, 'triplets')
    if args.save_predictions:
        os.makedirs(pred_dir, exist_ok=True)
    if args.save_triplets:
        os.makedirs(triplet_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    if device.type == 'cuda':
        print('gpu:', torch.cuda.get_device_name(0))

    dataset = CustomSegDataset(
        img_dir=args.test_images,
        mask_dir=args.test_masks,
        transform=val_aug,
        mode='test',
        img_suffix='.tif',
        mask_suffix='.tif',
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    print('test samples:', len(dataset))

    model = load_model(args.checkpoint, device)
    hist = torch.zeros(len(CLASSES), len(CLASSES), device=device)
    mask_lookup = {CustomSegDataset.normalize_id(mask_id + '.tif'): mask_id for _, mask_id in dataset.img_ids}

    for batch in tqdm(loader, desc='test'):
        images = batch['img'].to(device, non_blocking=True)
        masks = batch['gt_semantic_seg'].to(device, non_blocking=True)
        logits = model(images)
        logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        preds = logits.argmax(dim=1)
        hist += confusion_matrix(preds, masks, len(CLASSES))

        preds_np = preds.cpu().numpy().astype(np.uint8)
        for pred, image_id in zip(preds_np, batch['img_id']):
            if args.save_predictions:
                Image.fromarray(pred).save(osp.join(pred_dir, image_id + '_pred_ids.png'))
                Image.fromarray(mask_to_rgb(pred)).save(osp.join(pred_dir, image_id + '_pred_rgb.png'))
            if args.save_triplets:
                image_path = osp.join(args.test_images, image_id + '.tif')
                normalized_id = CustomSegDataset.normalize_id(image_id + '.tif')
                mask_id = mask_lookup[normalized_id]
                mask_path = osp.join(args.test_masks, mask_id + '.tif')
                save_triplet(image_path, mask_path, pred, osp.join(triplet_dir, image_id + '_triplet.png'))

    metrics = metrics_from_hist(hist)
    write_confusion_matrix(hist, args.output_dir)

    print('test mIoU:', metrics['mIoU'])
    print('test mF1:', metrics['mF1'])
    print('test OA:', metrics['OA'])
    lines = [
        'test mIoU: {:.6f}'.format(metrics['mIoU']),
        'test mF1: {:.6f}'.format(metrics['mF1']),
        'test OA: {:.6f}'.format(metrics['OA']),
        '',
        'per-class metrics:',
    ]
    for name, iou, f1 in zip(CLASSES, metrics['IoU'], metrics['F1']):
        line = '{} | IoU: {:.6f} | F1: {:.6f}'.format(name, float(iou), float(f1))
        print(line)
        lines.append(line)
    with open(osp.join(args.output_dir, 'test_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
