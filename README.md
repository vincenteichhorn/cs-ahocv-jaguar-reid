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

## Experiments
1. Embedding Model: MegaDescriptor, DINOv3, EfficientNet, MobileNet v3
2. Loss Functions: ArcFace, SubCenter ArcFace, CosFace, Triplet Loss (needs mining strategy, we use P-K Sampling)
3. Combined Losses: ArcFace + Triplet, CosFace + Triplet
4. Sampling Strategies: Random, Class-Balanced, Hard Negative Mining, P-K Sampling
5. Class Balanced Loss
6. Data Augmentation: geometric, color, cutour, mixup, cutmix
7. Optimizers: Adam, SGD, AdamW
8. Learning Rate Schedulers: StepLR, CosineAnnealingLR, OneCycleLR
9. Model Architectures: Projection Head variations, Dropout variations