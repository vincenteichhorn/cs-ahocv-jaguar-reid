# Leaderboard Experiments

## LBE01: Multiple Backbones

**WANDB Link:** <https://wandb.ai/josef-and-vincent/jaguar-reid-josefandvincent/groups/lbe01_multiple_backbones/overview>
**Research question:** How does the choice of frozen backbone architecture affect retrieval performance (mAP) and convergence stability in the Re-ID pipeline?
**Defined intervention::** Systematically vary the frozen backbone (DINOv3, EVA02, MegaDescriptor, EfficientNetB4, and MobileNetV3) while keeping the embedding projection, ArcFace loss, and optimizer fixed to the baseline.
**Defined intervention:** Track validation mAP and validation loss trajectories across epochs.
**Interpretation:** DINOv3 achieved the highest validation mAP ($\approx$ 0.86) and lowest loss, proving that high-capacity feature extractors benefit significantly from diverse pre-training. DINOv3 is selected for subsequent experiments due to its stable latent space separation.

## LBE02: Trained Backbone Pooling

**WANDB Link:** <https://wandb.ai/josef-and-vincent/jaguar-reid-josefandvincent/groups/lbe02_backbone_pooling/overview>
**Research question:** Does replacing standard global average pooling with Generalized Mean (GeM) pooling on top of a DINOv3 foundation model enhance discriminative embeddings?
**Defined intervention::** The baseline's MegaDescriptor linear head is replaced with DINOv3, a learnable GeM pooling layer, and Batch Normalization, keeping the ArcFace criterion and AdamW optimizer fixed.
**Defined intervention:** Monitor validation mAP over 100 epochs, comparing the DINOv3 + GeM configuration to the baseline.
**Interpretation:** The DINOv3 + GeM setup significantly outperformed the baseline and achieved a higher plateau, with Batch Normalization effectively mitigating covariate shift. The GeM parameter stabilized near 3, indicating a focus on more intense activations.

## LBE03: Projection Head Architecture

**WANDB Link:** <https://wandb.ai/josef-and-vincent/jaguar-reid-josefandvincent/groups/lbe03_projection_head_architecture/overview>
**Research question:** Does increasing the depth and width of the EmbeddingProjection head improve the discriminative power of embeddings from a frozen DINOv3 backbone?
**Defined intervention::** Vary the number of layers $n \in \{2, 3, 4\}$ and hidden dimensions $\in \{256, 512, 1024\}$ while keeping the final output dimension fixed at 256 with an ArcFace criterion.
**Defined intervention:** Compare training loss descent rates and final validation loss across all depth/width configurations.
**Interpretation:** Shallow, wider architectures (2 layers, 512 width) converged significantly faster and outperformed deeper models, which exhibited signs of optimization bottlenecks. A moderate depth of 2 layers will be fixed moving forward to balance capacity and generalization.

## LBE04: Data Augmentation

**WANDB Link:** <https://wandb.ai/josef-and-vincent/jaguar-reid-josefandvincent/groups/lbe04a_augmentation_frozen/overview> and <https://wandb.ai/josef-and-vincent/jaguar-reid-josefandvincent/groups/lbe04a_augmentation_trainable/overview>
**Research question:** Does introducing complex geometric transformations and regional erasures improve generalization on segmented jaguar images compared to the baseline?
**Defined intervention::** Compare four augmentation strategies (random affine/erasing, elastic rotation/perspective, complex mimic, and combined) across two states: a frozen DINOv3 backbone versus a fully trainable stack.
**Defined intervention:** Evaluate validation mAP over epochs to measure convergence speed, peak performance, and robustness.
**Interpretation:** Unfreezing the backbone yielded superior performance (mAP $> 0.88$), with random affine and erasing strategies providing the fastest convergence. Complex geometric "camera mimicry" introduced unnecessary variance, so future training should pair trainable backbones with moderate affine-erasing augmentations.

## LBE05: Hyperbolic vs. Hyperspherical Embedding Spaces

**WANDB Link:** <https://wandb.ai/josef-and-vincent/jaguar-reid-josefandvincent/groups/lbe05_spaces/overview>
**Research question:** Does transitioning the embedding space geometry from Euclidean to Hyperspherical or Hyperbolic manifolds affect identity-balanced mAP?
**Defined intervention::** Compare a Euclidean baseline against Hyperspherical and Hyperbolic projections (ablating curvature $\kappa$ and maxnorm clipping radii) using a frozen DINOv3 backbone.
**Defined intervention:** Track validation mAP and observe numerical stability under different space geometries and parameters.
**Interpretation:** While Hyperbolic space achieved the peak mAP (0.8860), it suffered from extreme numerical instability ("Hyperbolic Cliff") if clipping radii or curvature were aggressive. We will retain the Euclidean baseline to ensure training stability and minimize hyperparameters.

## LBE06: Loss Functions

**WANDB Link:** <https://wandb.ai/josef-and-vincent/jaguar-reid-josefandvincent/groups/lbe06a_loss_functions/overview> and <https://wandb.ai/josef-and-vincent/jaguar-reid-josefandvincent/groups/lbe06b_class_balanced_loss_functions/overview>
**Research question:** Do angular margin-based objectives, especially with class-balancing techniques, outperform traditional triplet-based approaches?
**Defined intervention::** Systematically replace the optimization criterion (Triplet, ArcFace, CosFace, Sub-Center ArcFace, Focal ArcFace, and their Class-Balanced variants) while maintaining a fixed DINOv3 backbone and projection head.
**Defined intervention:** Assess peak validation mAP and epochs required for convergence across the different formulations.
**Interpretation:** Class-Balanced ArcFace and Focal ArcFace proved most effective, converging significantly faster than Triplet Loss due to a more stable global proxy-based gradient. We will adopt Focal ArcFace moving forward to manage intra-class variance and handle class imbalance.

## LBE07: Progressive Resizing

**WANDB Link:** <https://wandb.ai/josef-and-vincent/jaguar-reid-josefandvincent/groups/lbe07_resizing/overview>
**Research question:** Can progressive image resizing during training improve identity-balanced mAP based on the curriculum learning hypothesis?
**Defined intervention::** Compare three fixed-resolution baselines against a 3-stage ($128 \rightarrow 256 \rightarrow 384$) and a 2-stage ($256 \rightarrow 384$) progressive schedule, fixing the MegaDescriptor backbone and ArcFace criterion.
**Defined intervention:** Evaluate final validation mAP at the maximum $384 \times 384$ resolution within a 100-epoch budget.
**Interpretation:** The fixed high-resolution baseline achieved the highest mAP (0.7955). Progressive resizing caused inefficient "re-learning" phases during transitions, increasing training time without local minimum benefits; it will not be used in the final model.

## LBE08: Optimizer Comparison

**WANDB Link:** <https://wandb.ai/josef-and-vincent/jaguar-reid-josefandvincent/groups/lbe08_optimizer/overview>
**Research question:** Which optimizer yields the highest identity-balanced mAP and the most stable convergence?
**Defined intervention::** Evaluate five optimization algorithms (Adam, AdamW, SGD, SGD with Nesterov, and RMSprop) across five random seeds using the identical baseline setup.
**Defined intervention:** Quantify stability (Divergence Rate below 0.75 mAP at epoch 20), mean validation mAP, and epochs required for convergence.
**Interpretation:** Adaptive optimizers (Adam, AdamW, RMSprop) provided 0% divergence and stable early-stage convergence, heavily outperforming SGD which suffered an 80% divergence rate and took 4.5x longer. We will use AdamW for the final model due to its high performance and robust reliability.

## LBE09: Learning Rate Scheduler Comparison

**WANDB Link:** <https://wandb.ai/josef-and-vincent/jaguar-reid-josefandvincent/groups/lbe09_lr_scheduler/overview>
**Research question:** Which learning rate scheduler yields the best identity-balanced mAP?
**Defined intervention::** Compare six scheduling strategies (StepLR, CosineAnnealingLR, OneCycleLR, ExponentialLR, ReduceLROnPlateau, and None) across five random seeds, fixing all other parameters.
**Defined intervention:** Track validation mAP, divergence rate, and convergence behavior over epochs.
**Interpretation:** OneCycleLR achieved the highest mAP (0.8879 $\pm$ 0.0134) via "super-convergence", stabilizing early gradient updates during warm-up. However, ReduceLROnPlateau will be selected for the final architecture as it demonstrated faster convergence and superior practical results.

## LBE10: Extensive Hyperparameter Search

**WANDB Link:** <https://wandb.ai/josef-and-vincent/jaguar-reid-josefandvincent/groups/lbe10_hyperparameter_search/overview>
**Research question:** Can a Bayesian search over architectural and loss-function hyperparameters significantly improve mAP compared to a static baseline?
**Defined intervention::** Transition to DINOv3, Focal ArcFace, segmented backgrounds, and heavy augmentation; implement a differential backbone learning rate; conduct a 30-trial Bayesian optimization sweep.
**Defined intervention:** Maximize validation mAP across a multi-dimensional search space (LRs, margins, scales, dropout, etc.).
**Interpretation:** Fine-grained tuning of the focal loss parameters and the balance between the backbone's learning rate and ArcFace margin allows convergence on robust embeddings. These optimal hyperparameters will be fixed to evaluate varying image resolutions for final deployment.

## LBE11: Performance Stability and Random Seed Analysis

**WANDB Link:** <https://wandb.ai/josef-and-vincent/jaguar-reid-josefandvincent/groups/lbe11_stability/overview>
**Research question:** To what extent does the model's performance depend on stochastic initialization, and is the observed mAP statistically robust across different random seeds?
**Defined intervention::** Execute the finalized pipeline (DINOv3, Focal ArcFace, inference refinements, optimized hyperparameters) across ten distinct random seeds while keeping all parameters fixed.
**Defined intervention:** Calculate the mean, standard deviation, and standard error of the validation mAP across all ten runs to quantify stability.
**Interpretation:** The model yielded a high mean mAP of 0.9379 with a remarkably low standard deviation ($\sigma = 0.0063$), demonstrating that the architecture is highly stable and seed-agnostic. This stable configuration will be frozen as the final submission candidate.

## LBE12: Inference Refinements

**WANDB Link:** <https://wandb.ai/josef-and-vincent/jaguar-reid-josefandvincent/groups/lbe12_inference_refinements/overview>
**Research question:** Can retrieval accuracy be maximized at inference time using feature space manifold updates, reciprocal relationship verification, and spatial invariance techniques?
**Defined intervention::** Implement a post-processing stack containing Query Expansion (QE), k-reciprocal re-ranking (Jaccard distance integration), and Test-Time Augmentation (TTA, via horizontal flipping).
**Defined intervention:** Assess the combined impact of these techniques on suppressing false positives and improving the final similarity rankings.
**Interpretation:** Query expansion effectively updates query representations toward cluster centers, while reciprocal re-ranking heavily suppresses false positives lacking mutual neighborhoods. Ensembling through TTA generates a highly robust, pose-invariant global signature.

## Final Experiment

**Kaggle Notebook:** <https://www.kaggle.com/code/vincenteichhorn03/final-jaguar>
**Kaggle Challenge:** <https://www.kaggle.com/competitions/jaguar-re-id>
**WANDB Link:** <https://wandb.ai/josef-and-vincent/jaguar-reid-josefandvincent/groups/final/overview>
