
# Me-owch! Automated feline pain detection using facial landmarks and machine learning
*Final-year BSc Computer Science project for COMP3000 Computing Project, University of Plymouth, 2025/2026.*

*Supervised by Dr Lauren Ansell.*

This repository contains a proof-of-concept pipeline for automated feline acute pain screening from cat face images. A ResNet-18 convolutional neural network predicts facial landmarks from a cropped cat face image. These landmarks are then used to compute Feline Grimace Scale-inspired geometric features and classify the image into one of four pain-likelihood buckets.

**Important**: This is a research prototype, not a clinical diagnostic tool. If you suspect a cat is in pain, contact a veterinary surgeon right away.

## Quick Start
The Gradio demo can be run from a fresh clone without the full CatFLW dataset.

### 1. Create and activate a virtual environment.
On Windows PowerShell:
```
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
Python 3.13 is recommended for a quick start. Python 3.11-3.13 should be suitable for PyTorch compatibility.

### 2. Install dependencies

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Run the demo

```
python -m src.app
```

The app should run a local Gradio interface. Upload a close-up cropped cat face image, or use one of the included example images.

## Important disclaimer

This project is intended to demonstrate a machine-learning pipeline for estimating whether a cat face image shows pain-associated facial features. It is designed for academic evaluation and proof-of-concept experimentation only.

It should not be used to diagnose pain, replace veterinary judgment, or make decisions about animal treatment. The model may be inaccurate, especially when images are poorly cropped, side-on, blurred, occluded, or of breeds with unusual facial morphology.

## Project overview

The project combines landmark detection with interpretable geometric measurements inspired by the Feline Grimace Scale.

The pipeline is:
```
Cropped cat face image
->
ResNet-18 landmark predictor
->
48 predicted facial landmarks
->
Geometric feature extraction
->
Nearest-distribution z-score voting
->
Pain-likelihood bucket
```

The four computed geometric features are:
1. Ear tips-to-bases ratio
2. Eye height-to-width ratio
3. Medial ear angle
4. Lateral ear angle

These features are compared against published control and painful thresholds from Evangelista *et al.* (2019). Each valid feature votes for the nearest distribution. The final output bucket is based on the proportion of features voting as painful:

* very_unlikely
* unlikely
* likely
* very_likely

## Repository structure
```
├── cat_model_resnet18.pt
├── requirements.txt
├── README.md
├── data/
│   └── example_crops/
├── docs/
├── notebooks/
│   └── results.ipynb
├── pain_labels/
├── training_histories/
└── src/
    ├── app.py
    ├── model_loader.py
    ├── cat_cnn.py
    ├── computations.py
    ├── pain_classifier.py
    ├── facial_landmark_labeller.py
    ├── nme.py
    └── tools.py
```

### Key files
|File or folder|Purpose|
|---|---|
|`src/app.py`|Gradio demo application.|
|`src/model_loader.py`| Loads the ResNet-18 architecture and checkpoint for the demo without importing the CatFLW-dependent training script.|
|`src/cat_cnn.py`|Dataset, training, evaluation, and model comparison code. Requires CatFLW in `data/images/` and `data/labels/`.|
|`src/computations.py`|Computes the four geometric features used by the pain classifier.|
|`src/pain_classifier.py`|Converts geometric features into pain-likelihood buckets using z-score voting.|
|`src/nme.py`|Normalised Mean Error evaluation for predicted landmarks. Requires CatFLW.|
|`notebooks/results.ipynb`|Results and evaluation notebook. CatFLW-dependent cells are skipped when the dataset is absent.|
|`data/example_crops/`|Example cropped cat images for the demo.|
|`training_histories/`|Saved training histories, including comparison models where available.|
|`pain_labels/`|Hand-labelled evaluation data used for pain-classification evaluation.|

## Model checkpoints
The final demo uses `cat_model_resnet18.pt`. Only the ResNet-18 model is required to run the final Gradio application. 

`cat_model_resnet50.pt` was omitted from the repository because of the file size. You can choose to train it by setting `run_training=True` and `model_name="resnet50"` in `cat_cnn.py`. Training histories are retained in `training_histories/` so ResNet-50 model behaviour such as overfitting can still be inspected.

The results notebook is designed to handle missing comparison checkpoints by skipping unavailable checkpoints.

## Installation
The minimal project requirements are in `requirements.txt` and include:
* torch
* torchvision
* opencv-python
* numpy
* gradio
* pandas
* matplotlib
* scikit-learn

To install on Windows PowerShell:
```
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### CUDA note
CUDA is optional. The Gradio demo can run on CPU. CUDA is mainly for faster training and evaluation. The code uses `torch.cuda.is_available()`. If CUDA is unavailable, the project falls back to CPU. 

If CUDA support is required, install the CUDA-enabled PyTorch build using the official PyTorch installation selector for your system and CUDA version.

## Running the Gradio demo
Run `python -m src.app`.

The app loads `cat_model_resnet18.pt`, predicts landmarks, computes the four geometric features, and displays:
* an annotated image with predicted landmarks.
* the final pain-likelihood bucket.
* a per-feature table showing feature values, z-scores, and votes.

The app does not require the full CatFLW dataset. The app imports `src/model_loader.py` instead of `src/cat_cnn.py` because `src/cat_cnn.py` sets up CatFLW paths at import time and therefore does not work without the CatFLW dataset.

## Data requirements by workflow

|Workflow|Requires CatFLW?|Required files/folders|Notes|
|---|---|---|---|
|Run Gradio demo|No|`cat_model_resnet18.pt`, `data/example_crops/`, `src/`|Works from a fresh repo clone.|
|Use example images|No|`data/example_crops/`|Included for demonstration.|
|Pain-label evaluation|No|`pain_labels/`, `notebooks/results.ipynb`|Uses the hand-labelled evaluation set included in the repo.|
|Run full landmark NME evaluation|Yes|`data/images/`, `data/labels/`|Requires CatFLW dataset.|
|Run train/validation/test split checks|Yes|`data/images/`, `data/labels/`|Requires CatFLW dataset.|
|Compare predicted vs. ground-truth landmarks|Yes|`data/images/`, `data/labels/`|Requires CatFLW dataset.|
|Retrain models|Yes|`data/images/`, `data/labels/`|Requires CatFLW dataset.|

## Dataset notes
This project uses the CatFLW dataset with 2,079 annotated cat images, 48 facial landmarks per image, and bounding boxes for face cropping. (Martvel *et al.*, 2023)

CatFLW is not included in this repository because it is a third-party dataset and very large. It is available [here](https://www.kaggle.com/datasets/georgemartvel/catflw). To reproduce the CatFLW-dependent training and evaluation, place the dataset files in the data folder as follows:
```
data/
├── images/
│   └── ...
└── labels/
    └── ...
```
The helper function in `src/tools.py` matches label filenames to images using common image extensions.

## Reproducing evaluation

The evaluation notebooks used throughout the dissertation have been consolidated into `notebooks/results.ipynb`. 

### Using the notebook without CatFLW
The notebook can be run without the full dataset. In this mode, CatFLW-dependent NME cells are skipped, training split checks are skipped, predicted-vs-ground-truth landmark comparison is skipped, and pain-label metrics can still run where the required local files are present.

### Using the notebook with CatFLW
If CatFLW is placed in `data/images/` and `data/labels/`, then the notebook can run the full landmark evaluation workflow, including train/validation/test split checks, landmark NME evaluation, model comparison where checkpoints are available, and predicted-vs-ground-truth landmark comparisons.

## Retraining models

The training code is in `src/cat_cnn.py`. To run it, CatFLW is required to be present in `data/images/` and `data/labels/`. The default model is ResNet-18. The implemented models are:
* resnet18
* resnet50
* efficientnet_b0
* mobilenet_v3_small

Training is disabled by default using `run_training = False`. To retrain, set `run_training = True` inside `src/cat_cnn.py` after ensuring CatFLW is present. CUDA is recommended for training, but the code can fall back to CPU.

## Results (summary)

The final selected model is ResNet-18. Model comparison included:

* ResNet-18
* ResNet-50
* EfficientNet-B0
* MobileNetV3-Small

Reported ResNet-18 landmark NME (normalised with bounding box diagonal):

|Region|NME|
|---|---|
|Overall|1.27%|
|Ears|2.06%|
|Eyes|0.96%|

Pain-classification evaluation used a 40-image non-expert hand-labelled evaluation set.

|Metric|Result|
|---|---|
|4-class accuracy|57%|
|Linear-weighted Cohen's kappa|0.53|
|Binary accuracy|77.5%|
|Binary Cohen's kappa|0.54|
|Sensitivity|63.2%|
|Specificity|90.5%|

These results should be interpreted as proof-of-concept evidence, not clinical validation.

## Limitations
The main limitation is the absence of a veterinary-grade dataset containing images labelled by confirmed pain state. As a result, the final pain classifier was evaluated against non-expert hand labels rather than veterinary diagnoses.

Other limitations include:
* CatFLW provides facial landmarks, not pain labels.
* The geometric features are approximations inspired by Feline Grimace Scale action units.
* Landmark errors can propagate into ratio and angle calculations.
* The model expects a close-up cropped cat face with eyes, ears, and muzzle visible.
* Performance may degrade on unusual poses, occlusions, poor lighting, blurred images, or flat-faced breeds.
* The classifier uses external published distributions and simple nearest-distribution voting rather than a clinically trained pain classifier.

## References
Evangelista, M.C. et al. (2019) ‘Facial expressions of pain in cats: the development and validation of a Feline Grimace Scale’, Scientific Reports, 9(1), p. 19128. Available at: https://doi.org/10.1038/s41598-019-55693-8.

Martvel, G. et al. (2023) ‘CatFLW: Cat Facial Landmarks in the Wild Dataset’. arXiv. Available at: https://doi.org/10.48550/arXiv.2305.04232.

## Licence and disclaimer
This repository contains code for academic assessment. This project is for research and educational use only and is not intended for clinical or commercial veterinary diagnosis. If you suspect a cat is in pain, please take it to a veterinary surgeon as soon as possible.

The publicly-available Kaggle dataset being used is the [Cat Facial Landmarks in the Wild (CatFLW)](https://www.kaggle.com/datasets/georgemartvel/catflw) dataset. It has 48 cat facial landmarks and bounding boxes for each of the cats' faces.

The dataset is being used for this project under the [Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) licence.

To reproduce the model training, please obtain the dataset from Kaggle and place it in the `/data` folder.