# Jaguar ReId

## Environment
Create a `.env` file in the root directory with the following variables:
```
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_api_key
```

```
poetry install
source $(poetry env info --path)/bin/activate
```

## Cluster
```
salloc --partition gpu-interactive --account=sci-demelo-computer-vision --cpus-per-task=32 --mem=100G --gpus=1 --time=08:00:00
```

## Experiments

## C1 Domain Understanding
- EX01: **Explanatory Dataset Analysis** class distribution, image quality, intra-class variability, inter-class similarity
  - Identity Distribution Plot
  - Viewpoint Classification?
  - Segmentation Quality Audit
  - Intra-class Variability Analysis
  - Inter-class Similarity Analysis
  - Near Duplicate Detection
  - Embedding Space Exploration
  - Saliency Map Visualization / Grad-CAM

## C2 Model Architecture
- EX02: **Embedding Model** MegaDescriptor, DINOv3, EfficientNet, MobileNet v3
- EX03: **Loss Functions** ArcFace, SubCenter ArcFace, CosFace, Triplet Loss, Focal Loss
- EX04: **Class Balanced Loss**
- EX05: **Model Architectures** GeM Pooling + BatchNorm1d + Projection Head
- EX06: **Projection Head Variations** different deep, different widths

## C3 Model Combinations
- EX07 **Fine-tuning** the embedding model and not the projection head

## C4 Data and Training Strategies
- EX08: **Sampling Strategies** Random, Class-Balanced, Hard Negative Mining, P-K Sampling
- EX09: **Data Augmentation** geometric, color, cutour, mixup, cutmix
- EX10: **Cross Validation** k-fold cross-validation, stratified k-fold cross-validation, leave-one-out cross-validation
- EX11: **Test Time Augmentation (TTA)**

## C5 Hyperparameter Optimization
- EX12: **Optimizers** Adam, SGD, AdamW
- EX13: **Learning Rate Schedulers** StepLR, CosineAnnealingLR, OneCycleLR
- EX14: **Basic Hyperparameters** batch size, learning rate, weight decay, momentum
