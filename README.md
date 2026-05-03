
# Me-owch! Automated feline pain detection using facial landmarks and machine learning
*Final-year BSc Computer Science project for COMP3000 Computing Project, University of Plymouth, 2025/2026.*

*Supervised by Dr. Lauren Ansell.*

This repository contains a proof-of-concept pipeline for automated feline acute pain screening from cat face images. A ResNet-18 convolutional neural network predicts facial landmarks from a cropped cat face image. These landmarks are then used to compute Feline Grimace Scale-inspired geometric features and classify the image into one of four pain-likeklihood buckets.

**Important**: This is a research prototype, not a clinical diagnostic tool. If you suspect a cat is in pain, contact a veterinary surgeon right away.

## Quick Start
The Gradio demo can be run from a fresh clone without the full CatFLW dataset.

### 1. Create and activate a virtual environment.
On Windows Powershell:
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
python -m scr.app
```

The app should run a local Gradio interface. Upload a close-up cropped cat face image, or use one of the included example images.

## Important disclaimer

This project is intended to demonstrate a machine-learning pipeline for estimating whether a cat face image shows pain-associated facial features. It is designed for academic evaluation and proof-of-concept experimentation only.

It should not be used to diagnose pain, replace veterinary judgment, or make decisions about animal treatment. The model may be inaccurate, especially when images are poorly cropped, side-on, blurred, occluded, or of breeds with unusual facial morphology.

## Project overview

The project combines landmark detection with interpretable geometric measurements inspired by the Feline Grimace Scale.

The pipeline is:
```
Cropped cat face image inputted
->
Resnet-18 landmark predictor
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
## Repository structure
```
├── cat_model_resnet18.pt
├── requirements.txt
├── README.md
├── data/
│ └── example_crops/
├── docs/
├── notebooks/
│ └── results.ipynb
├── pain_labels/ ├── training_histories/
└── src/
├── app.py
├── model_loader.py
├── cat_cnn.py
├── computations.py
├── pain_classifier.py
├── facial_landmark_labeller.py
├── nme.py
├── tools.py
└── pain_labeller.py
```

### Key files
|File or folder|Purpose|
|---|---|
|`src/app.py`|Gradio demo application.|
|`src/model_loader.py`| Loads the ResNet-18 architecture and checkpoint for the demo without importing the CatFLW-dependent training script.|
|`src/cat_cnn.py`|Dataset, training, evaluation, and model comparison code. Requires CatFLW in data/images/ and data/labels/.|
|`src/computations.py`|Computes the four geometric features used by the pain classifier.|
|`src/pain_classifier.py`|Converts geometric features into pain-likelihood buckets using z-score voting.|
|`src/nme.py`|Normalised Mean Error evaluation for predicted landmarks. Requires CatFLW.|
|`notebooks/results.ipynb`|Results and evaluation notebook. CatFLW-dependent cells are skipped when the dataset is absent.|
|`data/example_crops/`|Example cropped cat images for the demo.|
|`training_histories/`|Saved training histories, including comparison models where available.|
|`pain_labels/`|Hand labelled evaluation data used for pain-classification evaluation.|

## Model checkpoints
The final demo uses `cat_model_resnet18.pt`. Only the ResNet-18 model is required to run the final Gradio application. 

`cat_model_resnet50.pt` was omitted from the repository because of the file size. You can choose to train it by setting `run_training=True` and `model_name="resnet50"` in `cat_cnn.py`. Training histories are retained in `training_histories/` so ResNet-50 model behaviour such as overfitting can still be inspected.

## References
Caroli, P. (2022) Lean Inception, martinfowler.com. Available at: https://martinfowler.com/articles/lean-inception/ (Accessed: 18 October 2025).

Evangelista, M. (2018) ‘Facial expressions of pain in cats: development of the Feline Grimace Scale’, in ResearchGate. Available at: https://www.researchgate.net/publication/323830301_Facial_expressions_of_pain_in_cats_development_of_the_Feline_Grimace_Scale (Accessed: 13 October 2025).

Competition and Markets Authority (2025) Major reforms would require vet businesses to make fundamental changes to the way they support pet owners, GOV.UK. Available at: https://www.gov.uk/government/news/major-reforms-would-require-vet-businesses-to-make-fundamental-changes-to-the-way-they-support-pet-owners (Accessed: 15 October 2025).

Downing, R. and Della Rocca, G. (2023) ‘Pain in Pets: Beyond Physiology’, Animals : an Open Access Journal from MDPI, 13(3), p. 355. Available at: https://doi.org/10.3390/ani13030355.

Gruen, M.E. et al. (2022) ‘2022 AAHA Pain Management Guidelines for Dogs and Cats’, Journal of the American Animal Hospital Association, 58(2), pp. 55–76. Available at: https://doi.org/10.5326/JAAHA-MS-7292.

Horwitz, D.F. and Rodan, I. (2018) ‘Behavioral awareness in the feline consultation: Understanding physical and emotional health’, Journal of Feline Medicine and Surgery, 20(5), pp. 423–436. Available at: https://doi.org/10.1177/1098612X18771204.

McNamee, B.S. et al. (2025) Vets should be made to publish prices, competition watchdog says, BBC News. Available at: https://www.bbc.co.uk/news/articles/c201r14z6r3o (Accessed: 15 October 2025).

Mills, D.S. et al. (2020) ‘Pain and Problem Behavior in Cats and Dogs’, Animals, 10(2), p. 318. Available at: https://doi.org/10.3390/ani10020318.

Mundschau, V. and Suchak, M. (2023) ‘When and Why Cats Are Returned to Shelters’, Animals : an Open Access Journal from MDPI, 13(2), p. 243. Available at: https://doi.org/10.3390/ani13020243.

University of Plymouth (2025) Risk Assessment Form (RA1). University of Plymouth. Available at: https://www.psy.plymouth.ac.uk/home/Documents/RA-LinkLabs.pdf (Accessed: 13 October 2025).
