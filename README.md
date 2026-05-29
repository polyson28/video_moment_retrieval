# Negative-Aware Video Moment Retrieval With Query-Aware Compression

This repository contains the experimental code for a diploma project on open-set video moment retrieval:

> Given a video and a text query, retrieve the relevant temporal moment when the query is positive, and reject the query when the queried moment is absent.

The project studies the trade-off between:

- video-token compression;
- temporal localization quality;
- reject-or-retrieve quality;
- inference efficiency.

The main final method is `QA-TopK` query-aware compression with a learned reject classifier. The older `QA-TopK + context + diversity reserve` variant is kept as a context/reserve ablation, not as the main joint result.

## Repository Layout

```text
diploma_project/
  compression/
    query_aware.py
  data_layer/
    schemas.py
    manifest_builder.py
    dataset.py
    validate_manifest.py
  eval/
    metrics.py
    bootstrap.py
  experiments/
    baseline_trainable_full_token_retrieval_head.py
    joint_compression_reject.py
    joint_reject_type_ablation.py
    joint_context_reserve_ablation.py
    final_protocol_additions.py

notebooks/
  data_analysis.ipynb

Makefile
requirements.txt
AGENTS.md
```

Generated files are written to:

```text
results/   # intermediate experiment outputs, predictions, trained retrieval head
outputs/   # final diploma tables and figures
data/      # local datasets and features, not committed
```

## Data Sources

This project uses QVHighlights and Moment of Untruth splits.

QVHighlights:

- Original paper/code: <https://github.com/jayleicn/moment_detr>
- Moment-DETR repository says it contains QVHighlights annotations and provides pre-extracted features.
- Moment-DETR feature archive: `moment_detr_features.tar.gz`, linked from the official repository README.
- Raw videos are not required for the main experiments in this repository.

Moment of Untruth:

- Code/splits: <https://github.com/keflanagan/MomentofUntruth>
- Paper: "Moment of Untruth: Dealing with Negative Queries in Video Moment Retrieval", WACV 2025.
- The repository provides negative-query files for positive-only, in-domain negative, out-of-domain negative, and mixed settings.

The expected local directory structure is:

```text
data/
  qvhighlights/
    annotations/
      highlight_train_release.jsonl
      highlight_val_release.jsonl
      highlight_test_release.jsonl
      subs_train.jsonl
    features/
      clip_features/
        <vid>.npz
      clip_text_features/
        qid<qid>.npz
      slowfast_features/
      pann_features/
      clip_sub_features/

  qvhighlights_neg/
    pos_only/
      qvhl_pos_train.jsonl
      qvhl_pos_val.jsonl
    indomain/
      qvhl_id_neg_train.jsonl
      qvhl_id_neg_val.jsonl
      qvhl_id_pos_neg_train.jsonl
      qvhl_id_pos_neg_val.jsonl
    outofdomain/
      qvhl_ood_neg_train.jsonl
      qvhl_ood_neg_val.jsonl
      qvhl_ood_pos_neg_train.jsonl
      qvhl_ood_pos_neg_val.jsonl
    ood_negative_sentences.jsonl
```

The code assumes Moment-DETR-style `.npz` feature files:

- video features: `data/qvhighlights/features/clip_features/<vid>.npz`, array key `features`;
- text features: `data/qvhighlights/features/clip_text_features/qid<source_qid>.npz`, array key `pooler_output`.

## Environment

Python 3.12 was used during development.

Install dependencies:

```bash
make install
```

Equivalent command:

```bash
python -m pip install -r requirements.txt
```

Run a syntax/import sanity check:

```bash
make check
```

This runs:

```bash
python -m compileall diploma_project
```

## Full Py-Only Reproduction

After the data are placed under `data/`, run:

```bash
make all-results
```

This executes the full pipeline in the required order:

```bash
python -m diploma_project.experiments.baseline_trainable_full_token_retrieval_head
python -m diploma_project.experiments.joint_compression_reject
python -m diploma_project.experiments.joint_reject_type_ablation
python -m diploma_project.experiments.joint_context_reserve_ablation
python -m diploma_project.experiments.final_protocol_additions --lightweight
python -c "from diploma_project.experiments.final_protocol_additions import make_fixed_window_control; make_fixed_window_control(run_inference=True)"
```

The full run creates:

```text
results/exp1b_trainable_full_token_retrieval_head/
results/exp6_joint_compression_reject/
results/exp6b_joint_reject_type_ablation/
results/exp6c_joint_context_reserve_ablation/
outputs/tables/
outputs/figures/
```

`make all-results` can take a while because it scores train and validation splits several times. The fixed-window control is especially slow.

## Makefile Targets

### `make exp1b`

Runs:

```bash
python -m diploma_project.experiments.baseline_trainable_full_token_retrieval_head
```

This is the required positive-only retrieval baseline used by all later joint experiments.

What is trained:

- `SGDRegressor` window-level IoU ranker;
- input features are handcrafted temporal-window statistics over CLIP token-query similarity;
- train split: positive-only QVHighlights train;
- validation split: positive-only QVHighlights val;
- no compression;
- no reject head;
- no negative queries.

Candidate window lengths:

```text
[2, 4, 8, 16, 32]
```

Main outputs:

```text
results/exp1b_trainable_full_token_retrieval_head/model.joblib
results/exp1b_trainable_full_token_retrieval_head/scaler.joblib
results/exp1b_trainable_full_token_retrieval_head/metrics.json
results/exp1b_trainable_full_token_retrieval_head/predictions_val_pos.jsonl
```

The files `model.joblib` and `scaler.joblib` are required by every later experiment.

Additional outputs:

```text
results/exp1b_trainable_full_token_retrieval_head/config.yaml
results/exp1b_trainable_full_token_retrieval_head/manifest.json
results/exp1b_trainable_full_token_retrieval_head/threshold.json
results/exp1b_trainable_full_token_retrieval_head/notes.md
results/exp1b_trainable_full_token_retrieval_head/best_cases_val_pos.csv
results/exp1b_trainable_full_token_retrieval_head/worst_cases_val_pos.csv
results/exp1b_trainable_full_token_retrieval_head/metrics_by_video_length.csv
results/exp1b_trainable_full_token_retrieval_head/metrics_by_gt_moment_length.csv
results/exp1b_trainable_full_token_retrieval_head/pred_window_length_tokens_distribution.csv
results/exp1b_trainable_full_token_retrieval_head/comparison_exp1_vs_exp1b.csv
```

Main metrics:

- `R1@0.3`
- `R1@0.5`
- `R1@0.7`
- `mean_iou`
- predicted window length diagnostics

### `make joint`

Runs:

```bash
python -m diploma_project.experiments.joint_compression_reject
```

What is evaluated/trained:

- uses the retrieval head from `results/exp1b_trainable_full_token_retrieval_head/`;
- scores positive, ID-negative, and OOD-negative queries;
- trains a learned reject classifier for systems with `confidence_classifier`;
- calibrates reject thresholds on a held-out subset of `train_mixed`;
- reports on separate `val_pos`, `val_id_neg`, `val_ood_neg`.

Compression settings:

- no compression;
- uniform compression at budgets `0.5`, `0.25`, `0.1`;
- query-aware compression at budgets `0.5`, `0.25`, `0.1`.

The main query-aware policy in this module is:

```text
QA-TopK, no context, no diversity reserve
```

Outputs:

```text
results/exp6_joint_compression_reject/config.yaml
results/exp6_joint_compression_reject/manifest.json
results/exp6_joint_compression_reject/metrics.json
results/exp6_joint_compression_reject/notes.md
results/exp6_joint_compression_reject/metrics/summary_by_system.csv
results/exp6_joint_compression_reject/metrics/summary_by_system.json
results/exp6_joint_compression_reject/metrics/val_pos_metrics.csv
results/exp6_joint_compression_reject/metrics/val_id_neg_metrics.csv
results/exp6_joint_compression_reject/metrics/val_ood_neg_metrics.csv
results/exp6_joint_compression_reject/metrics/threshold_sensitivity.csv
results/exp6_joint_compression_reject/predictions/train_predictions.csv
results/exp6_joint_compression_reject/predictions/val_predictions.csv
results/exp6_joint_compression_reject/reject/reject_features_train.csv
results/exp6_joint_compression_reject/reject/reject_features_val.csv
results/exp6_joint_compression_reject/reject/thresholds.json
results/exp6_joint_compression_reject/figures/
```

Main metrics:

- `R1@0.5_before_reject`
- `R1@0.7_before_reject`
- `R1@0.5_e2e`
- `R1@0.7_e2e`
- `mean_iou`
- `RA_ID`
- `RA_OOD`
- `BalancedOpenSet@0.5`
- `compression_ratio`
- `avg_inference_time_per_query`

### `make reject-ablation`

Runs:

```bash
python -m diploma_project.experiments.joint_reject_type_ablation
```

What is compared:

- no reject;
- top-window-score threshold;
- learned confidence classifier.

Important score-scale convention:

- learned classifier uses `predict_proba[:, 1]` as `confidence_prob`;
- top-score threshold uses `top_window_score`;
- these two score sources are not mixed.

Outputs:

```text
results/exp6b_joint_reject_type_ablation/config.yaml
results/exp6b_joint_reject_type_ablation/manifest.json
results/exp6b_joint_reject_type_ablation/metrics.json
results/exp6b_joint_reject_type_ablation/notes.md
results/exp6b_joint_reject_type_ablation/metrics/summary_by_system.csv
results/exp6b_joint_reject_type_ablation/metrics/summary_by_system.json
results/exp6b_joint_reject_type_ablation/metrics/val_pos_metrics.csv
results/exp6b_joint_reject_type_ablation/metrics/val_id_neg_metrics.csv
results/exp6b_joint_reject_type_ablation/metrics/val_ood_neg_metrics.csv
results/exp6b_joint_reject_type_ablation/predictions/train_predictions.csv
results/exp6b_joint_reject_type_ablation/predictions/val_predictions.csv
results/exp6b_joint_reject_type_ablation/reject/reject_features_train.csv
results/exp6b_joint_reject_type_ablation/reject/reject_features_val.csv
results/exp6b_joint_reject_type_ablation/reject/thresholds.json
results/exp6b_joint_reject_type_ablation/figures/
```

This experiment is also used by:

```text
outputs/tables/confidence_scale_diagnostics.csv
```

### `make context-ablation`

Runs:

```bash
python -m diploma_project.experiments.joint_context_reserve_ablation
```

What is compared:

- `qa_topk`;
- `qa_topk_context`;
- `qa_topk_context_diversity_r0p10`;
- `qa_topk_context_diversity_r0p20`.

Each variant is evaluated with and without learned reject, at budgets:

```text
0.5, 0.25, 0.1
```

Outputs:

```text
results/exp6c_joint_context_reserve_ablation/config.yaml
results/exp6c_joint_context_reserve_ablation/manifest.json
results/exp6c_joint_context_reserve_ablation/metrics.json
results/exp6c_joint_context_reserve_ablation/notes.md
results/exp6c_joint_context_reserve_ablation/metrics/summary_by_system.csv
results/exp6c_joint_context_reserve_ablation/metrics/summary_by_system.json
results/exp6c_joint_context_reserve_ablation/metrics/summary_by_variant.csv
results/exp6c_joint_context_reserve_ablation/metrics/val_pos_metrics.csv
results/exp6c_joint_context_reserve_ablation/metrics/val_id_neg_metrics.csv
results/exp6c_joint_context_reserve_ablation/metrics/val_ood_neg_metrics.csv
results/exp6c_joint_context_reserve_ablation/predictions/train_predictions.csv
results/exp6c_joint_context_reserve_ablation/predictions/val_predictions.csv
results/exp6c_joint_context_reserve_ablation/reject/reject_features_train.csv
results/exp6c_joint_context_reserve_ablation/reject/reject_features_val.csv
results/exp6c_joint_context_reserve_ablation/reject/thresholds.json
results/exp6c_joint_context_reserve_ablation/figures/
```

This experiment is important because the final diploma table uses `QA-TopK` rows from:

```text
results/exp6c_joint_context_reserve_ablation/metrics/summary_by_variant.csv
```

### `make final-outputs`

Runs:

```bash
python -m diploma_project.experiments.final_protocol_additions --lightweight
```

This command collects the final advisor-response tables and figures from the experiment folders under `results/`.

It creates:

```text
outputs/tables/joint_main_qatopk.csv
outputs/tables/joint_main_qatopk_bootstrap_ci.csv
outputs/tables/joint_main_qatopk_by_seed.csv
outputs/tables/joint_main_qatopk_seed_summary.csv
outputs/tables/ood_triviality.csv
outputs/tables/confidence_scale_diagnostics.csv
outputs/tables/nonadaptive_extra_baselines.csv
outputs/tables/window_length_distribution.csv
outputs/figures/query_length_by_type.png
outputs/figures/predicted_window_length_distribution.png
```

It does not fully recompute:

```text
outputs/tables/fixed_window_control.csv
```

Use `make fixed-window` for that.

### `make fixed-window`

Runs:

```bash
python -c "from diploma_project.experiments.final_protocol_additions import make_fixed_window_control; make_fixed_window_control(run_inference=True)"
```

This is the long-window-bias control.

It first measures the baseline full-retrieval predicted window length distribution, then recomputes retrieval/open-set evaluation with fixed candidate window sets:

```text
[8]
[16]
[32]
[4, 8, 16]
[4, 8, 16, 32]
```

Systems:

```text
Full retrieval
Uniform beta=0.5
QA-TopK beta=0.5
QA-TopK beta=0.25
```

Outputs:

```text
outputs/tables/window_length_distribution.csv
outputs/tables/fixed_window_control.csv
outputs/figures/predicted_window_length_distribution.png
```

## Final Output Tables

The final diploma-facing tables are in `outputs/tables/`.

### `joint_main_qatopk.csv`

Main joint table.

Rows:

- Full retrieval no reject;
- Full retrieval + learned reject;
- Uniform compression + learned reject, beta `0.5`, `0.25`, `0.1`;
- QA-TopK compression no reject, beta `0.5`, `0.25`, `0.1`;
- QA-TopK compression + learned reject, beta `0.5`, `0.25`, `0.1`.

Important columns:

- `R1@0.5_e2e`
- `R1@0.7_e2e`
- `RA_ID`
- `RA_OOD`
- `BalancedOpenSet`
- `BalancedOpenSet_ID`
- `compression_ratio`
- `avg_inference_time_per_query`

### `joint_main_qatopk_bootstrap_ci.csv`

Query-level bootstrap confidence intervals.

Metrics:

- `R1@0.3`
- `R1@0.5`
- `R1@0.7`
- `mean_iou`
- `R1@0.5_e2e`
- `RA_ID`
- `RA_OOD`
- `BalancedOpenSet`
- `BalancedOpenSet_ID`

### `joint_main_qatopk_by_seed.csv`

Per-seed table for seeds:

```text
42, 43, 44
```

The rerun changes:

- train/calibration split for reject;
- classifier training seed;
- random components where applicable.

### `joint_main_qatopk_seed_summary.csv`

Seed-level aggregate with:

- `metric_mean`
- `metric_std`
- `metric_ci_low`
- `metric_ci_high`

### `ood_triviality.csv`

OOD diagnostic table.

Checks:

- `source_qid` overlap between positive and OOD;
- query length statistics for `pos`, `id_neg`, `ood_neg`;
- whether `RA_OOD = 1.0` for reject configurations.

The corresponding figure is:

```text
outputs/figures/query_length_by_type.png
```

### `confidence_scale_diagnostics.csv`

Confidence-score scale audit.

Columns include:

- `notebook`
- `system`
- `reject_type`
- `confidence_source`
- `threshold_min`
- `threshold_max`
- `score_min`
- `score_max`

Interpretation:

- `confidence_source = predict_proba[:, 1]` means learned classifier probability;
- `confidence_source = top_window_score` means raw top retrieval window score.

### `window_length_distribution.csv`

Distribution of predicted window lengths for baseline full retrieval.

Used to diagnose long-window bias.

### `fixed_window_control.csv`

Fixed-window control table.

Metrics:

- `R1@0.5`
- `mean_iou`
- `R1@0.5_e2e`
- `RA_ID`
- `RA_OOD`
- `BalancedOpenSet`
- `BalancedOpenSet_ID`

The corresponding figure is:

```text
outputs/figures/predicted_window_length_distribution.png
```

### `nonadaptive_extra_baselines.csv`

Extra non-adaptive baselines, when available from earlier positive-only compression runs:

- random keep;
- video-only saliency.

These are not the main final protocol; they are reported as extra context.

## Metrics

Positive retrieval metrics:

- `R1@0.3`
- `R1@0.5`
- `R1@0.7`
- `mean_iou`

End-to-end retrieval metric:

```text
R1@0.5_e2e = positive query is accepted AND IoU >= 0.5
```

Reject metrics:

```text
RA_ID  = fraction of ID-negative queries rejected
RA_OOD = fraction of OOD-negative queries rejected
```

Balanced open-set metrics:

```text
BalancedOpenSet    = (R1@0.5_e2e + RA_ID + RA_OOD) / 3
BalancedOpenSet_ID = (R1@0.5_e2e + RA_ID) / 2
```

`BalancedOpenSet_ID` is important because the OOD split is very easy in the current experiments: OOD reject accuracy is often exactly `1.0`.

Efficiency metrics:

- `retained_token_ratio`
- `compression_ratio`
- `avg_inference_time_per_query`
- `approx_attention_cost_ratio`

## Reject Threshold Protocol

For learned reject systems:

1. score all train/validation examples with the retrieval head and compression policy;
2. split `train_mixed` into fit and calibration subsets;
3. train `LogisticRegression(class_weight="balanced")` on reject features;
4. use `predict_proba[:, 1]` as `confidence_prob`;
5. choose a threshold on the held-out train calibration subset;
6. evaluate on `val_pos`, `val_id_neg`, `val_ood_neg`.

Decision rule:

```text
reject if confidence_prob < threshold
```

For top-score threshold ablation:

```text
reject if top_window_score < threshold
```

Top-window score and classifier probability are different score scales and are intentionally kept separate.

## Notes On Reproducibility

The pipeline is deterministic for fixed seeds where scikit-learn and NumPy are deterministic. The fixed seeds used in the final seed analysis are:

```text
42, 43, 44
```

The generated `results/` and `outputs/` directories can be deleted and recreated with:

```bash
make all-results
```

If only final tables need to be recreated from existing `results/`, run:

```bash
make final-outputs
make fixed-window
```

If only the main retrieval head is missing, run:

```bash
make exp1b
```
