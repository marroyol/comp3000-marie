import os
import json

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from src.facial_landmark_labeller import LANDMARK_INDEX_MAP
from src.tools import find_matching_image

EAR_INDICES = sorted(i for i, name in LANDMARK_INDEX_MAP.items() if "ear" in name)
EYE_INDICES = sorted(i for i, name in LANDMARK_INDEX_MAP.items() if "eye" in name)

def evaluate_nme(model, label_files, image_dir, label_dir, device, img_size=224):
    model.eval()
    to_tensor = T.ToTensor()

    with open(os.path.join(label_dir, label_files[0])) as f:
        first_annotation = json.load(f)
    num_landmarks = len(first_annotation["labels"])

    other_indices = sorted(
        i for i in range(num_landmarks)
        if i not in EAR_INDICES and i not in EYE_INDICES
    )

    sum_nme_per_landmark = np.zeros(num_landmarks, dtype=np.float64)
    count_per_landmark = np.zeros(num_landmarks, dtype=np.int64)

    with torch.no_grad():
        for label_file in label_files:
            label_path = os.path.join(label_dir, label_file)
            with open(label_path) as f:
                annotation = json.load(f)

            true_landmarks = np.array(annotation["labels"], dtype = np.float32)

            x_min, y_min, x_max, y_max = annotation["bounding_boxes"]
            x_min, y_min = int(x_min), int(y_min)
            x_max, y_max = int(x_max), int(y_max)
            crop_w = x_max - x_min
            crop_h = y_max - y_min

            if crop_w <=0 or crop_h <=0:
                continue

            bbox_diagonal = float(np.sqrt(crop_w ** 2 + crop_h ** 2))

            img_path = find_matching_image(label_file, image_dir)
            if img_path is None:
                continue
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            cropped = image_rgb[y_min:y_max, x_min:x_max]
            if cropped.size == 0:
                continue

            resized = cv2.resize(cropped, (img_size, img_size))
            image_tensor = to_tensor(resized).unsqueeze(0).to(device)

            pred_normalised = model(image_tensor).cpu().view(-1, 2).numpy()

            true_in_crop = true_landmarks.copy()
            true_in_crop[:, 0] -= x_min
            true_in_crop[:, 1] -= y_min

            true_normalised = true_in_crop.copy()
            true_normalised[:,0] /= crop_w
            true_normalised[:,1] /= crop_h

            dx_px = (pred_normalised[:, 0] - true_normalised[:, 0]) * crop_w
            dy_px = (pred_normalised[:, 1] - true_normalised[:, 1]) * crop_h
            pixel_errors = np.sqrt(dx_px ** 2 + dy_px ** 2)
            nme_per_landmark_this_image = pixel_errors / bbox_diagonal
            
            sum_nme_per_landmark += nme_per_landmark_this_image
            count_per_landmark += 1

    num_images = int(count_per_landmark[0]) if count_per_landmark[0] >0 else 0
    if num_images == 0:
        raise RuntimeError("For evaluate_nme, no images were successfully evaluated")
    
    per_landmark_percent = (sum_nme_per_landmark / count_per_landmark) * 100.0

    def group_mean(indices):
        if not indices:
            return float("nan")
        return float(per_landmark_percent[indices].mean())
    
    return {
        "overall": float(per_landmark_percent.mean()),
        "ears": group_mean(EAR_INDICES),
        "eyes": group_mean(EYE_INDICES),
        "other": group_mean(other_indices),
        "per_landmark": per_landmark_percent,
        "num_images": num_images
    }

def print_nme_report(results, model_name=None):
    header = "NME Report"
    if model_name:
        header += f" for {model_name}"
    print("\n" + header)
    print("=" * len(header))
    print(f"Images evaluated: {results['num_images']}")
    print(f"Normaliser: bounding box diagonal")
    print()
    print(f"  Overall NME: {results['overall']:.3f}%")
    print(f"  Ears NME: {results['ears']:.3f}%   "
          f"(landmarks {EAR_INDICES})")
    print(f"  Eyes NME: {results['eyes']:.3f}%   "
          f"(landmarks {EYE_INDICES})")
    print(f"  Other NME: {results['other']:.3f}%")
    print()
 
    per_lm = results["per_landmark"]
    worst = np.argsort(per_lm)[::-1][:5]
    print("5 worst landmarks (index : NME%):")
    for idx in worst:
        name = LANDMARK_INDEX_MAP.get(int(idx), "(undefined)")
        print(f"{int(idx):3d} {name:30s} {per_lm[idx]:.3f}%")
    print()
 