# Exploratory Data Analysis Experiments
## EDA01: Dataset Characteristics Overview
**Research question:** We hypothesize that the dataset follows a long-tail distribution characteristic of wildlife monitoring, and that brightness, contrast, and sharpness will adhere to normal distributions.
**Defined intervention:** This is an exploratory analysis rather than a model intervention; it evaluates image metrics (sharpness, contrast, brightness, background ratio) and identity frequencies using an 80/20 stratified split.
**Defined intervention:** Quantify underlying image quality metrics and the distribution of identities across the training and validation splits
**Interpretation:** The dataset suffers from a pronounced class imbalance and right-skewed sharpness, risking model bias toward popular identities Moving forward, specific loss functions or balancing strategies must be considered to prevent minority class degradation.

## EDA02: Multiple Background Interventions
**WANDB Link:** https://wandb.ai/josef-and-vincent/jaguar-reid-josefandvincent/groups/eda02_background_interventions/overview
**Research question:** Does environmental context facilitate or bias the re-identification process, and will removing background force the model to learn more robust subject-centric features?
**Defined intervention:** The baseline DINOv3 backbone and ArcFace criterion remain fixed, while image backgrounds are systematically varied using five protocols: blurred, segmented (black), noisy, random, and untouched
**Defined intervention:** Measure Mean Average Precision (mAP) and validation loss over 100 epochs for each background intervention.
**Interpretation:** The untouched background achieved the highest validation mAP ($\approx$ 0.86), indicating that the original environmental context provides significant auxiliary information. We must account for the model's reliance on spatial environmental cues in future iterations.

## EDA03: Duplicates/Near-Duplicates and Intra- and Inter-Class-Similarity
**Research question:** Can a frozen foundation model effectively distinguish between intra-class identities and inter-class samples using cosine similarity?
**Defined intervention:** The MegaDescriptor model weights are fixed while systematically varying the cosine similarity decision threshold $\tau\in\{0.90,0.95,0.99\}$ to assess pairwise matching performance
**Defined intervention:** Utilize Receiver Operating Characteristic (ROC) analysis and False Discovery Rate (FDR) curves to evaluate matching.
**Interpretation:** The pre-trained space is not fully optimized for the dataset's specific variance, as higher thresholds eliminated false positives but caused significant recall loss. Near-duplicates represent hard negatives and should not be removed; future work must utilize a metric learning phase like ArcFace

## EDA04: Curated Dataset 
**WANDB Link:** https://wandb.ai/josef-and-vincent/jaguar-reid-josefandvincent/groups/eda04_curated/overview
**Research question:** Can a principled, multi-stage curation pipeline reduce the training set size while preserving or improving the re-identification performance?
**Defined intervention:** The dataset is reduced via a four-stage curation pipeline (near-duplicate removal, low-quality filtering, outlier pruning, and over-representation capping) and evaluated against the full dataset using a frozen MegaDescriptor baseline.
**Defined intervention:** Train two identical models (one on the full dataset, one on the curated dataset) and compare validation mAP, training time, and epochs until convergence.
**Interpretation:** While curation reduced training time by 48.3% and improved data quality, it caused a significant mAP degradation (-0.0668) because the model became data-starved. For small-scale datasets, training on the full dataset remains the stronger approach.
