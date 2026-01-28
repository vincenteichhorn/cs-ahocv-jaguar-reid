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
1. C2 Embedding Model: MegaDescriptor, DINOv3, EfficientNet, MobileNet v3
2. C2 Loss Functions: ArcFace, SubCenter ArcFace, CosFace, Triplet Loss (needs mining strategy, we use P-K Sampling)
3. C2 Combined Losses: ArcFace + Triplet, CosFace + Triplet
4. C2 Class Balanced Loss
5. C4 Sampling Strategies: Random, Class-Balanced, Hard Negative Mining, P-K Sampling
6. C4 Data Augmentation: geometric, color, cutour, mixup, cutmix
7. C5 Optimizers: Adam, SGD, AdamW

8. C5 Learning Rate Schedulers: StepLR, CosineAnnealingLR, OneCycleLR
9. C2 Model Architectures: Projection Head variations, Dropout variations
10. C4 Cross Validation: k-fold cross-validation, stratified k-fold cross-validation, leave-one-out cross-validation
11. C4 Test Time Augmentation (TTA)
12. C3 Fine-tuning the embedding model and not the projection head
13. C1 Explanatory Dataset Analysis: class distribution, image quality, intra-class variability, inter-class similarity
14. C5 Basic Hyperparameters: batch size, learning rate, weight decay, momentum
