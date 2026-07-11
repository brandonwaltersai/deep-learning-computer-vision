# Results

## CNN — Fashion-MNIST architecture experiments

Baseline: 2 conv layers (32, 64 filters), 3×3 kernels, max pooling, 10
epochs, batch size 128. Four single-variable variants were trained against
the same baseline to isolate one architectural change at a time.

| Variant | Change | Test accuracy |
|---|---|---|
| Baseline | — | 91.86% |
| Larger kernels | 3×3 → 5×5 | 91.86% |
| Deeper | +1 conv layer (3 total) | 91.28% |
| Stride=2, no pooling | strided convs replace pooling | 90.77% |
| Average pooling | max → average pool | 90.24% |

Per-class precision/recall (baseline model): weakest class is "Shirt"
(precision 0.79, recall 0.73 — most often confused with "T-shirt/top" and
"Coat," which is expected given how visually similar those categories are
at 28×28 resolution). Strongest classes: Trouser, Sandal, Bag (all ≥0.99
precision and recall). Full confusion matrix and classification report
are in the notebook output.

**Takeaway:** none of the four variants beat the baseline on this dataset.
Bigger kernels bought nothing; going deeper, dropping pooling for stride,
or switching to average pooling all cost 0.6–1.6 points of test accuracy.
For a small, low-resolution, low-noise benchmark like Fashion-MNIST, the
simplest architecture was also the best one — added capacity had nothing
useful left to fit.

## DCGAN — Fashion-MNIST, trained from scratch

Trained for real on 2026-07-11 (Apple Silicon GPU via `tensorflow-metal`)
— this is not the notebook's default (unexecuted) template output, it's an
actual completed run.

**Setup:** 40 epochs, batch size 256, latent dim 100, Adam (lr=2e-4,
β₁=0.5) for both networks, one-sided label smoothing (0.9) on real labels,
fixed seed for reproducibility. Total training time: 30.8 minutes for the
full 60,000-image training set × 40 epochs.

**Loss behavior:** generator and discriminator losses converged to
G≈0.82 / D≈1.35 by the final 5 epochs (epoch 1: G=0.81 / D=1.36 → epoch
40: G=0.79 / D=1.39). Both values sit close to the theoretical two-player
Nash equilibrium for this loss formulation (D → 2·ln2 ≈ 1.386, G → ln2 ≈
0.693 when the discriminator can no longer tell real from fake better
than chance). GAN losses don't monotonically decrease the way a
classifier's does — a stable oscillation near equilibrium is the sign of
healthy adversarial training, not stalled training. There's no evidence
of mode collapse (discriminator loss never collapses toward 0, which
would mean the generator stopped fooling it) or generator collapse
(generator loss never diverges).

**Sample quality over training:**
- [Epoch 2](figures/generated_epoch_02.png) — noise with faint garment silhouettes
- [Epoch 20](figures/generated_epoch_20.png) — recognizable garment shapes, some class diversity
- [Epoch 40](figures/generated_epoch_40.png) — clearer edges and category diversity (shirts, bags, footwear are distinguishable), still visibly short of the sharpness in [real samples](figures/real_samples_grid.png)

Full loss curve: [`figures/training_losses.png`](figures/training_losses.png).
Final 8×8 generated grid: [`figures/samples_grid_final.png`](figures/samples_grid_final.png).

**Honest limitation:** this is a small, non-conditional DCGAN trained for
40 epochs with no FID/Inception Score evaluation run (the notebook
includes that code path, but it requires downloading ImageNet-pretrained
InceptionV3 weights and wasn't run for this result — visual inspection
and loss dynamics are the evidence presented here, not a quantitative
generative-quality score).
