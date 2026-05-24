import argparse
import os
import os.path as osp
from glob import glob

import matplotlib.cm as cm
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from segmentation_models_pytorch.encoders import get_preprocessing_fn


ROOT = osp.dirname(osp.abspath(__file__))
CLASSES = (
    'background', 'hazelnut', 'forest', 'permanent_cropland', 'greenhouse',
    'grassland', 'sparsely_vegetated_areas', 'arable_land',
    'discontinuous_urban_fabric', 'road_and_rail_networks', 'water_courses',
    'water_bodies', 'wetland',
)


def normalize(cam):
    cam = cam - cam.min()
    return cam / (cam.max() + 1e-8)


def overlay_heatmap(image, cam_map, alpha=0.45):
    image_np = np.asarray(image.convert('RGB')).astype(np.float32) / 255.0
    heatmap = cm.get_cmap('jet')(cam_map)[..., :3].astype(np.float32)
    overlay = image_np * (1.0 - alpha) + heatmap * alpha
    return Image.fromarray(np.uint8(np.clip(overlay, 0.0, 1.0) * 255))


def id_from_triplet(path):
    return osp.basename(path).replace('_triplet.png', '')


def crop_triplet(path):
    img = Image.open(path).convert('RGB')
    width, height = img.size
    return [img.crop((i * width // 3, 0, (i + 1) * width // 3, height)) for i in range(3)]


def preprocess(image, image_size):
    resized = image.convert('RGB').resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(resized).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def preprocess_with_encoder(image, image_size, preprocessing):
    resized = image.convert('RGB').resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(resized)
    if preprocessing is not None:
        arr = preprocessing(arr)
    else:
        arr = arr.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder = checkpoint.get('encoder', 'mit_b2')
    encoder_weights = checkpoint.get('encoder_weights', 'imagenet')
    image_size = checkpoint.get('image_size', 512)
    model = smp.Segformer(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=len(CLASSES)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    preprocessing = get_preprocessing_fn(encoder, pretrained=encoder_weights) if encoder_weights is not None else None
    return model, image_size, preprocessing


def require_segformer():
    if not hasattr(smp, 'Segformer'):
        raise RuntimeError(
            'This segmentation_models_pytorch installation does not provide smp.Segformer. '
            'Install a newer SegFormer-capable version, preferably in Python >= 3.9: '
            'python -m pip install -U segmentation-models-pytorch timm'
        )


def find_target_layer(model):
    decoder_convs = []
    all_convs = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            all_convs.append((name, module))
            if name.startswith('decoder') and module.out_channels > len(CLASSES):
                decoder_convs.append((name, module))
    if decoder_convs:
        return decoder_convs[-1]
    for name, module in reversed(all_convs):
        if 'segmentation_head' not in name and module.out_channels > len(CLASSES):
            return name, module
    return all_convs[-1]


class ActivationCapture:
    def __init__(self, layer):
        self.activation = None
        self.gradient = None
        self.handles = [
            layer.register_forward_hook(self.forward_hook),
            layer.register_full_backward_hook(self.backward_hook),
        ]

    def forward_hook(self, _module, _inputs, output):
        self.activation = output

    def backward_hook(self, _module, _grad_input, grad_output):
        self.gradient = grad_output[0]

    def close(self):
        for handle in self.handles:
            handle.remove()


def target_class_from_prediction(logits):
    pred = logits.argmax(dim=1)[0]
    counts = torch.bincount(pred.flatten(), minlength=len(CLASSES)).float()
    counts[0] = 0
    if counts.max() == 0:
        return int(torch.bincount(pred.flatten(), minlength=len(CLASSES)).argmax().item())
    return int(counts.argmax().item())


def target_score(logits, class_id):
    pred = logits.argmax(dim=1)[0]
    class_logits = logits[0, class_id]
    mask = pred == class_id
    if mask.sum() == 0:
        return class_logits.mean()
    return class_logits[mask].mean()


def gradcam(model, x, layer, class_id=None):
    capture = ActivationCapture(layer)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    if class_id is None:
        class_id = target_class_from_prediction(logits)
    score = target_score(logits, class_id)
    score.backward(retain_graph=True)
    activations = capture.activation.detach()
    gradients = capture.gradient.detach()
    capture.close()
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam_map = (weights * activations).sum(dim=1, keepdim=True)
    cam_map = F.relu(cam_map)
    cam_map = F.interpolate(cam_map, size=x.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
    return normalize(cam_map.cpu().numpy()), class_id


def gradcamplusplus(model, x, layer, class_id=None):
    capture = ActivationCapture(layer)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    if class_id is None:
        class_id = target_class_from_prediction(logits)
    score = target_score(logits, class_id)
    score.backward(retain_graph=True)
    activations = capture.activation.detach()
    gradients = capture.gradient.detach()
    capture.close()

    grads_power_2 = gradients.pow(2)
    grads_power_3 = gradients.pow(3)
    denominator = 2 * grads_power_2 + (activations * grads_power_3).sum(dim=(2, 3), keepdim=True)
    denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
    alpha = grads_power_2 / (denominator + 1e-8)
    weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
    cam_map = (weights * activations).sum(dim=1, keepdim=True)
    cam_map = F.relu(cam_map)
    cam_map = F.interpolate(cam_map, size=x.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
    return normalize(cam_map.cpu().numpy()), class_id


@torch.no_grad()
def eigencam(model, x, layer, class_id=None):
    activation = {}

    def hook(_module, _inputs, output):
        activation['value'] = output.detach()

    handle = layer.register_forward_hook(hook)
    logits = model(x)
    handle.remove()
    if class_id is None:
        class_id = target_class_from_prediction(logits)
    acts = activation['value'][0]
    channels, height, width = acts.shape
    flat = acts.reshape(channels, height * width).transpose(0, 1)
    flat = flat - flat.mean(dim=0, keepdim=True)
    try:
        _, _, v = torch.pca_lowrank(flat, q=1)
        cam_map = torch.matmul(flat, v[:, 0]).reshape(height, width)
    except Exception:
        u, _s, _v = torch.svd(flat)
        cam_map = u[:, 0].reshape(height, width)
    cam_map = F.relu(cam_map)
    cam_map = F.interpolate(cam_map[None, None], size=x.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
    return normalize(cam_map.cpu().numpy()), class_id


def save_panel(image, mask, overlay, class_id, method_label, image_id, out_path):
    image = image.convert('RGB').resize((360, 360), Image.BILINEAR)
    mask = mask.convert('RGB').resize((360, 360), Image.BILINEAR)
    overlay = overlay.convert('RGB').resize((360, 360), Image.BILINEAR)
    canvas = Image.new('RGB', (1120, 450), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype('arial.ttf', 22)
        font = ImageFont.truetype('arial.ttf', 18)
    except Exception:
        font_title = ImageFont.load_default()
        font = ImageFont.load_default()
    draw.text((20, 15), 'SegFormer | {} | {} | target: {}'.format(method_label, image_id, CLASSES[class_id]), fill=(0, 0, 0), font=font_title)
    for idx, (label, panel) in enumerate(zip(['Image', 'Ground Truth', 'XAI Heatmap Overlay'], [image, mask, overlay])):
        x = 20 + idx * 370
        canvas.paste(panel, (x, 60))
        draw.rectangle((x, 60, x + 360, 420), outline=(35, 35, 35), width=2)
        draw.text((x + 95, 425), label, fill=(0, 0, 0), font=font)
    canvas.save(out_path)


def collect_image_ids(triplet_dir, limit):
    ids = [id_from_triplet(p) for p in glob(osp.join(triplet_dir, '*_triplet.png'))]
    preferred = [
        'Area1_Image_65_41', 'Area1_Image_65_44', 'Area1_Image_62_43',
        'Area5_Image_16_93', 'Area2_Image_013_052', 'Area5_Image_22_93',
        'Area5_Image_29_73', 'Area2_Image_011_030', 'Area1_Image_47_64',
        'Area5_Image_24_96',
    ]
    selected = [x for x in preferred if x in ids]
    for image_id in ids:
        if image_id not in selected:
            selected.append(image_id)
        if len(selected) >= limit:
            break
    return selected[:limit]


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM, Grad-CAM++ and Eigen-CAM for trained SegFormer.')
    parser.add_argument('--checkpoint', default=osp.join(ROOT, 'best.pth'))
    parser.add_argument('--triplet-dir', default=osp.join(ROOT, 'test_results', 'triplets'))
    parser.add_argument('--output-root', default=osp.join(ROOT, 'xai_results'))
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--methods', nargs='+', default=['gradcam', 'gradcamplusplus', 'eigencam'])
    parser.add_argument('--target-class', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    require_segformer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    if device.type == 'cuda':
        print('gpu:', torch.cuda.get_device_name(0))

    model, image_size, preprocessing = load_model(args.checkpoint, device)
    layer_name, layer = find_target_layer(model)
    print('target layer:', layer_name)
    image_ids = collect_image_ids(args.triplet_dir, args.limit)
    if not image_ids:
        raise RuntimeError('No triplet images found. Run smp_segformer_test.py with --save-triplets first.')

    method_fns = {
        'gradcam': (gradcam, 'Grad-CAM'),
        'gradcamplusplus': (gradcamplusplus, 'Grad-CAM++'),
        'eigencam': (eigencam, 'Eigen-CAM'),
    }

    for image_id in image_ids:
        triplet_path = osp.join(args.triplet_dir, image_id + '_triplet.png')
        image, mask, _prediction = crop_triplet(triplet_path)
        x = preprocess_with_encoder(image, image_size, preprocessing).to(device)
        for method in args.methods:
            method_fn, method_label = method_fns[method]
            out_dir = osp.join(args.output_root, method)
            os.makedirs(out_dir, exist_ok=True)
            cam_map, class_id = method_fn(model, x, layer, args.target_class)
            cam_img = Image.fromarray(np.uint8(cam_map * 255)).resize(image.size, Image.BILINEAR)
            overlay = overlay_heatmap(image, np.asarray(cam_img).astype(np.float32) / 255.0)
            overlay.save(osp.join(out_dir, image_id + '_overlay.png'))
            cam_img.save(osp.join(out_dir, image_id + '_heatmap.png'))
            save_panel(image, mask, overlay, class_id, method_label, image_id, osp.join(out_dir, image_id + '_xai_panel.png'))
            print('{} | {} | target={}'.format(image_id, method_label, CLASSES[class_id]))


if __name__ == '__main__':
    main()
