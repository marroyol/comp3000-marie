import os

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import gradio as gr
import pandas as pd

from src.model_loader import model, device, model_path
from src.pain_classifier import classify_landmarks
from src.facial_landmark_labeller import LANDMARK_INDEX_MAP

# Load the trained landmark model once so each upload only runs inference.
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()
_to_tensor = T.ToTensor()

BUCKET_DISPLAY = {
    "very_unlikely": "Very unlikely to show pain-associated facial features",
    "unlikely": "Unlikely to show pain-associated facial features",
    "likely": "Likely to show pain-associated facial features",
    "very_likely": "Very likely to show pain-associated facial features"
}

def _run_pipeline(image_rgb):
    """Run image upload -> landmark prediction -> feature classification -> annotation"""
    h, w, _ = image_rgb.shape
    resized = cv2.resize(image_rgb, (224, 224))
    t = _to_tensor(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        # Reshape the 96 regression outputs into 48 landmark (x, y) pairs
        pred = model(t).cpu().view(-1, 2).numpy()

        # Convert normalised outputs back to uploaded-image pixel coordinates
        pred_px = pred.copy()
        pred_px[:,0] *= w
        pred_px[:,1] *= h

        # Classification uses the geometric features computed from pixel landmarks
        result = classify_landmarks(pred_px.tolist())

        annotated = image_rgb.copy()
        for i, (x,y) in enumerate(pred_px):
            xi, yi = int(round(x)), int(round(y))
            is_defined = i in LANDMARK_INDEX_MAP
            if is_defined:
                name = LANDMARK_INDEX_MAP[i]
                if "ear" in name:
                    colour = (255, 80, 80)
                elif "eye" in name:
                    colour = (80, 180, 255)
                else:
                    colour = (255, 255, 0)
                radius = 4
            else:
                colour = (140, 140, 140)
                radius = 2
            cv2.circle(annotated, (xi, yi), radius, colour, -1)
            cv2.circle(annotated, (xi, yi), radius + 1, (0, 0, 0), 1)
        
        return annotated, result
    
def analyse_cat(image):
    if image is None:
        return None, "Please upload a cat photo.", None
    
    annotated, result = _run_pipeline(image)

    bucket = result["bucket"]
    if bucket is None:
        headline = "Could not analyse this image."
    else:
        display = BUCKET_DISPLAY[bucket]
        frac = result["painful_fraction"]
        n_valid = result["n_valid_features"]
        headline = (f"## {display}\n\n"
                    f"**{int(round(frac * n_valid))} of {n_valid} features**"
                    f" vote painful ({frac:.0%}).")
    
    rows = []
    # These rows expose the same feature values and z-scores used for the bucket
    pretty = {
        "ear_tips_bases_ratio": "Ear tips / bases ratio",
        "eye_height_width_ratio": "Eye height / width ratio",
        "medial_ear_angle": "Medial ear angle in degrees",
        "lateral_ear_angle": "Lateral ear angle in degrees",
    }

    for feature_name, detail in result["details"].items():
        if detail["vote"] is None:
            rows.append([pretty[feature_name], "N/a", "-", "-", "skipped"])
        else:
            rows.append([
                pretty[feature_name],
                f"{detail['value']:.3f}",
                f"{detail['z_control']:.2f}",
                f"{detail['z_painful']:.2f}",
                detail["vote"]
            ])
    df = pd.DataFrame(rows, columns=[
        "Feature", "Value", "z (control)", "z (painful)", "Vote"
    ])
    
    return annotated, headline, df

DESCRIPTION = """

This is a research prototype for the University of Plymouth module COMP3000 undergraduate research dissertation. It is using a ResNet18 CNN trained on the CatFLW dataset (Martvel et al., 2023)
to detect feline facial landmarks, then computes four geometric features inspired by the Feline Grimace Scale (Evangelista et al., 2019) and classifies the image into one of four pain
likelihood buckets using externally derived thresholds.

**To use it, upload a close-up photo of a cat's face, filling most of the frame. The model expects to see both ears, both eyes, and the muzzle.**
"""

DISCLAIMER = """

**This tool is not a diagnostic tool! If you suspect your cat is in pain, please consult a veterinary surgeon!** Only registered veterinary surgeons can diagnose animals under the UK Veterinary
Surgeons Act 1966. This is a research demonstration and may be inaccurate, especially for flat-faced breeds (Persians, Scottish Folds, and similar).
"""

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_images_root = os.path.join(_base, "data", "example_crops")

_example_filenames = [
    "paz3.png",
    "00000451_018.png",
    "00000064_027.png",
    "CAT_01_00000110_021.png",
    "00000078_006.png",
    "00000322_007.png",
    "00000361_021.png",
]

_example_files = []
for fname in _example_filenames:
    path = os.path.join(_images_root, fname)
    if os.path.exists(path):
        _example_files.append([path])

with gr.Blocks(title="Automated feline pain detection using facial landmarks and machine learning") as demo:
    gr.Markdown(DESCRIPTION)
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Upload a front-facing cat photo with the face cropped",
                type="numpy",
                height=400
            )
            submit_button = gr.Button("Analyse", variant="primary")
            if _example_files:
                gr.Examples(
                    examples=_example_files,
                    inputs=input_image,
                    label="Or try an example! (click to load)",
                )
        
        with gr.Column():
            output_image = gr.Image(
                label="Predicted landmarks",
                height=400,
            )
            output_headline = gr.Markdown()
            output_table = gr.Dataframe(
                label="Per-feature breakdown",
                interactive=False
            )
        
    submit_button.click(
        fn=analyse_cat,
        inputs=[input_image],
        outputs=[output_image, output_headline, output_table],
    )

    gr.Markdown(
        "---\n"
        "Built with Gradio. ResNet18 model, CatFLW dataset. "
        "Pain classifier: nearest-class vote using Evangelista et al. (2019) Table 1."
    )

if __name__ == "__main__":
    demo.launch(share=False, debug=True)