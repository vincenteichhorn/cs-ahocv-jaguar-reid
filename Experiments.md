# Experiments
A valid experiment answers one clear research question using a systematic, controlled approach and includes:
- Research question or hypothesis (explicit, testable)
- Defined intervention (what changes vs baseline, and what stays fixed)
- Evaluation protocol (appropriate to the question)
- Interpretation (what you learned, why it happened, and what you would do next)


## EDA_EXPERIMENTS
### 1. Multiple Background Interventions
Question: Which Background intervention reduces context reliance without harming identity cues?

Interventions:
- Gray/Black
- Blur
- Noise
- Random

Document:
- Table of Interventions vs identity-balanced mAP
- Model + Training Protocol/Config across interventions
- Interpretation: what changes of reveal about reliance on background

### 2. Curated Dataset
Most unique, high-quality, representative samples for training

Document:
- Selection criteria and method
- mAP vs full dataset and efficiency gains
- Interpretation of what valuable samples look like

### 3. Deduplcateation Methods
Question: Which deduplication method best improves identity-balanced mAP?

Document:
- Definition of near-duplicate  (exact duplicate vs similairty threshold)
- Methosd tried (perceptual hashing, SSIM, embedding similarity, clustering, kNN pruning, hybrids)
- Scope (within-identity only vs across identities) and the reason
- mAP before and after, sensitifity to threshold
- Effects on training dynamics and mining behavior when relevant

## LEADERBOARD_EXPERIMENTS
### 1. Multiple Backbones
Question: Which backbone architecture provides the best identity-balanced mAP?

Backbones:
- MegaDescriptor
- DINOv3
- EfficientNet
- MobileNet
- EVA

Criteria:
- Same training protocol, loss, schedule, augmentation, evaluation
- same embedding dimension (or justification)
- mAP + efficiency metric (number of parameters, number of trainable parameters)

Document:
- Why these backbones?
- Table: mAP + efficiency metrics
- Interpretation: What characteristics matter and why?

### 2. Neural Architecture Search (NAS) to beat baseline mAP
Question: Can NAS find a better architecture than the baseline for identity-balanced mAP?

Document:
- Search space, algorithm, budget (number of trials or GPU-hours)
- Best model vs baseline and budget-matched baseline
- Patterns discovered

### 3. Extensive Hyperparameter Search
Question: Can hyperparameter tuning improve identity-balanced mAP?

Document:
- Search space, method, number of trials
- Best configuration vs baseline
- Hyperparameter importance or sensitivity analysis
- Use of learning rate scheduler

### 4. Data Augmentation
Question: Which data augmentations improve identity-balanced mAP?

Document:
- Hypothesis-driven study with controlled component ablations and insights

### 5. Hyperbolic vs hyperspherical Embedding spaces
Question: Does the choice of embedding space geometry affect identity-balanced mAP?

Document:
- Motivation and implementation details
- Controlled setup
- mAP and stability notes
- Interpretatiion: optional embedding diagnostics

### 6. Loss Functions
Question: Which loss function optimizes identity-balanced mAP best?

Requirement:
- Controlled comparison (same backbone, schedule, augmentations, embedding, dimension, evaluation)
- Report identity-balanced mAP and training stability notes
- Interpretation of why the better loss fits this dataset

Total 7 Loss Functions (3.5 points)

### 7. Progressive resizing
Question: Does progressive resizing improve identity-balanced mAP?

Document:
- Study of efficiency vs accuracy, robustness, or stability, with ablations

### 8. Best experiment according to mAP five to ten times with different random seeds
Question: How stable is the best experiment's performance?

Document:
- Exact configuration
- List of random seeds
- Identity-balanced mAP for each seed
- Mean and standard deviation across seeds
- Interpretation:
    - Standard deviation small --> results support claim of improvement
    - Standard deviation large --> results indicate instability and limits strength of claim

### 9. Compare different optimizers
Question: Which optimizer yields the best identity-balanced mAP?

Requirements:
- Controlled setup
- Clear definitions: optimizer hyperparameters
- Report identity-balanced mAP plus training stability indicators (divergence rate, variance across seeds, convergence curves)

Document:
- Comparison plan (which optimizers and why)
- Table of results
- Training dynamics: convergence speed, stability and sensitivity
- Mean and standard deviation across seeds for top contenders
- Interpretation: Why best choice fits this task (regularization, noisy gradients, batch size effects)

### 10. Compare different schedulers
Question: Which scheduler yields the best identity-balanced mAP?

Requirements:
- Controlled setup
- Clear definitions: scheduler hyperparameters
- Report identity-balanced mAP plus training stability indicators (divergence rate, variance across seeds, convergence curves)

Document:
- Comparison plan (which schedulers and why)
- Table of results
- Training dynamics: convergence speed, stability and sensitivity
- Mean and standard deviation across seeds for top contenders
- Interpretation: Why best choice fits this task (regularization, noisy gradients, batch size effects)