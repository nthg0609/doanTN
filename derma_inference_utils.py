import numpy as np
import cv2
import torch


def multi_scale_prob_map(image_rgb, model, device, scales=(1.0, 0.75, 0.5), input_size=224, mean=None, std=None):
    """Run model on multiple center crops/scales and return averaged probability map.

    image_rgb: HxWx3 uint8 RGB
    model: torch model returning logits or single-channel probability map
    mean/std: sequences of length 3 or None
    Returns: prob_map (HxW float32)
    """
    H, W = image_rgb.shape[:2]
    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for s in scales:
            # compute center crop for scale s
            if s >= 1.0:
                crop = image_rgb.copy()
                y1, x1, ch, cw = 0, 0, H, W
            else:
                ch = max(1, int(round(H * s)))
                cw = max(1, int(round(W * s)))
                y1 = (H - ch) // 2
                x1 = (W - cw) // 2
                crop = image_rgb[y1:y1 + ch, x1:x1 + cw]

            # resize crop to model input
            resized = cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
            arr = resized.astype(np.float32) / 255.0
            if mean is not None and std is not None:
                arr = (arr - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
            inp = np.transpose(arr, (2, 0, 1))[None, ...]
            t = torch.from_numpy(inp).to(device).float()

            out = model(t)
            # handle models that return tuple
            if isinstance(out, (tuple, list)):
                out = out[0]
            if isinstance(out, torch.Tensor):
                prob = torch.sigmoid(out).squeeze().cpu().numpy()
                # If multi-channel, take first
                if prob.ndim == 3:
                    prob = prob[0]
            else:
                # fallback if model returns numpy
                prob = np.array(out)

            prob = cv2.resize(prob, (cw, ch), interpolation=cv2.INTER_LINEAR)
            prob_map[y1:y1 + ch, x1:x1 + cw] += prob
            count_map[y1:y1 + ch, x1:x1 + cw] += 1.0

    avg = prob_map / np.maximum(count_map, 1e-6)
    return avg


def multiscale_segment_from_rgb(image_rgb, model, device, scales=(1.0, 0.75, 0.5), input_size=224, threshold=0.5, min_area_px=64, mean=None, std=None, morph_kernel=5):
    """Produce a binary mask from multi-scale averaged probabilities and simple postprocessing.

    Returns: mask (HxW uint8 0/1), prob_map (HxW float32), seg_info dict
    """
    prob_map = multi_scale_prob_map(image_rgb, model, device, scales=scales, input_size=input_size, mean=mean, std=std)
    bin_mask = (prob_map >= threshold).astype(np.uint8)

    if morph_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    final = np.zeros_like(bin_mask)
    lesions = 0
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area_px:
            final[labels == i] = 1
            lesions += 1

    seg_info = {
        'method': 'deeplab_multiscale',
        'threshold': float(threshold),
        'min_area_px': int(min_area_px),
        'scales': list(scales),
        'morph_kernel': int(morph_kernel),
        'lesion_found': int(lesions)
    }

    return final.astype(np.uint8), prob_map.astype(np.float32), seg_info
