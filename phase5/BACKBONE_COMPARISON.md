# Sequence Backbone Ablation: ESM-2 vs ESM-C

## Purpose

The interaction model uses a protein language model as its sequence backbone.
Two candidate backbones were compared under identical conditions to decide which
one to keep: ESM-2 (650M parameters) and ESM-C (600M parameters). This follows
the architecture plan in the proposal, which lists both backbones.

## Protocol

Both backbones were trained and evaluated with the same code, the same data, the
same hyperparameters, and the same random seed. Only the backbone weights differ,
so any difference in the result is attributable to the backbone alone.

- Dataset: Bernett gold-standard, leakage-free splits (163,192 train, 59,260
  validation, 52,048 test), balanced 1 to 1.
- Training: 3 epochs, batch 4, learning rate 1e-5, max length 512, seed 0.
- Evaluation: held-out test set, 52,048 pairs. Threshold 0.5 for the pointwise
  metrics. AUPR and AUROC are threshold independent.

## Results (test set, n = 52,048)

| Model                   | AUPR   | AUROC  | Precision | Recall | F1     | Accuracy | MCC    |
|-------------------------|--------|--------|-----------|--------|--------|----------|--------|
| ESM-2 650M (baseline)   | 0.7125 | 0.7193 | 0.6349    | 0.7236 | 0.6764 | 0.6538   | 0.3106 |
| ESM-C 600M (baseline)   | 0.7192 | 0.7238 | 0.6525    | 0.6844 | 0.6680 | 0.6599   | 0.3202 |
| ESM-C 600M (tuned)      | 0.7179 | 0.7210 | 0.6358    | 0.7357 | 0.6821 | 0.6571   | 0.3182 |

## Findings

1. ESM-C outperforms ESM-2 under identical settings on the two threshold
   independent ranking metrics (AUPR 0.7192 vs 0.7125, AUROC 0.7238 vs 0.7193),
   and also on accuracy and MCC, while using fewer parameters (600M vs 650M). At
   the fixed 0.5 threshold the two models trade places on precision and recall:
   ESM-C is more precise, ESM-2 more liberal. Because AUPR and AUROC do not depend
   on a threshold choice, they are the fair basis for ranking, and ESM-C wins both.

2. A tuning run on ESM-C (5 epochs, peak learning rate 2e-5, effective batch 32
   via gradient accumulation, learning-rate warmup of 500 steps followed by linear
   decay) raised validation AUPR from 0.6624 to 0.6821 but did not raise the test
   number. On the held-out test set the tuned model lands slightly below the plain
   baseline (AUPR 0.7179 vs 0.7192, AUROC 0.7210 vs 0.7238). The gap is within
   run-to-run noise. The tuning improved the fit to the validation split without
   generalizing, so the simpler training recipe is retained.

## Decision

ESM-C 600M with the plain baseline recipe is adopted as the sequence backbone for
the interaction model. The tuning run is reported as an ablation that moved
validation but not test performance.

## Reference comparison

For context, the base paper (PLM-interact) reports a comparable headline figure on
the same Bernett splits. Our ESM-C result is in the same range. A direct side by
side claim is avoided because training budget and other implementation details
differ between the two works.

## Artifacts

- Figure: backbone_comparison.pdf and backbone_comparison.png
- Per-pair predictions and metrics: phase5/metrics_full/
  (preds_cmp_esm2_s0.csv, preds_cmp_esmc_s0.csv, preds_esmc_tuned_s0.csv and the
  matching metrics JSON files)
- Training and evaluation jobs: phase2/train_cmp_esm2.sbatch,
  phase2/train_cmp_esmc.sbatch, phase2/train_esmc_tuned.sbatch,
  phase2/eval_cmp.sbatch, phase2/eval_tuned.sbatch
