# DTSA 5511 Final Project — Porto Seguro Safe Driver (Tabular DL)

**Problem.** Predict whether a customer will file an auto-insurance claim in the next year. Binary, **heavily imbalanced**.

**Data & provenance.** Kaggle: *Porto Seguro Safe Driver Prediction* (`train.csv`). Features are anonymized: `*_num`, `*_bin`, `*_cat`. Missing values often encoded as `-1`. I use the public training split only. License/terms as on Kaggle.

**Goal & metrics.** Optimize discrimination and ranking under class imbalance. I report **ROC-AUC**, **PR-AUC** (primary), Brier score (calibration), and confusion matrix at the **F1-optimal threshold** on a stratified hold-out set.

**EDA**
- Positive rate: ~3–4%.  
- Missingness: several columns encode missing as `-1` → I add **missing indicators** per numeric feature.  
- Numerics show skew; I **median-impute** and **standardize**.  
- Categoricals: high-cardinality in places; I **label-encode** with train-only vocab + **unknown** bucket for unseen.

**Methods.**
- **Baseline:** XGBoost (`gpu_hist` when available) with `scale_pos_weight`.
- **Deep model:** MLP for tabular with **categorical embeddings** + numeric block, dropout, BCE with `pos_weight` (and a focal-loss ablation). Early stopping on **PR-AUC**.  
- **Ablations:** (1) BCE vs Focal; (2) with vs without categorical embeddings; (3) class-balanced sampler on/off.  
- **Calibration:** reliability plot + Brier score.

**Validation.** Single stratified **80/20** split, `seed=42`. No time component in features, so plain split is acceptable. All preprocessing fit on train only.

**Repro.**
- Environment: Kaggle Notebook (CPU/GPU).  
- Data path: `/kaggle/input/porto-seguro-safe-driver-prediction/train.csv`.  
- Run cells top-to-bottom. Figures stored under `/kaggle/working/reports/figures/`. Summary CSV at `/kaggle/working/results_summary.csv`.  
- Random seeds fixed where supported.

**Notes / limitations.**
- Features are anonymized engineered signals; external covariates were not added.
- On tabular data, tree ensembles are strong baselines; I include both and discuss trade-offs.

**Academic honesty.** This is my own work. I used public documentation for library usage; all sources are cited where relevant.
