# Clearance Risk Engine — HS Code Recommender for Customs Clearance

End-to-end ML project on synthetic apparel shipment data. Predicts the correct customs HS (Harmonized System) code from a free-text product description, enabling proactive customer communication for international clearance.

> **Note:** This is a learning project on synthetic data, built to deepen hands-on ML skills. It is inspired by — but not affiliated with — any production system.

## Problem

International customs clearance breaks because consignors don't know which HS code applies to their product, and consignees don't know what documents are required at the destination. Today, problems get discovered when shipments hit customs and get held — at which point fixing them is expensive.

This project builds the ML core of a **proactive clearance assistant**: at booking time, given a product description, the system predicts the most likely HS code, looks up the documents and duty rate the consignee will need at destination, and triggers a personalized outreach message before the shipment moves.

## Architecture

The system has two intelligence layers:

1. **HS Code Recommender (ML)** — multi-class text classification predicting top-3 HS codes from product description with calibrated confidence scores
2. **Rules Engine (deterministic)** — lookup table indexed by `(hs_code, destination_country)` returning required documents, duty rate, and restrictions

This separation reflects production design: prediction where it's appropriate, deterministic lookups where regulation requires auditability.

## Dataset

**5000 synthetic apparel shipments** across 30 HS codes (Chapters 61 knitted + 62 woven), with rule-based product descriptions designed to mirror real-world consignor input variability. Consignor countries: IN, CN, BD, VN. Destination countries: US, UK, CA, DE.

**120 lookup rules** — one per (HS code, destination) combination, encoding required documents, duty rate, and restrictions per country regime (CBP HTSUS, UK Global Tariff, CBSA, EU TARIC).

## Methodology

### Feature engineering
- TF-IDF with unigrams + bigrams
- `min_df=2`, `max_df=0.95`, English stopwords removed
- Vocabulary size: ~940 features

### Models
- **Logistic Regression** — baseline, fast, interpretable, calibrated probabilities
- **Random Forest** — non-linear challenger with `class_weight='balanced'`

### Evaluation
- **Top-1 accuracy** — model's #1 prediction matches truth
- **Top-3 accuracy** — true class appears in model's top 3 (the metric that matters for the product)
- **Per-class precision/recall** — surface confused classes
- **Error analysis** — manual review of misclassifications

## Key results

| Model | Top-1 Accuracy | Top-3 Accuracy |
|---|---|---|
| Logistic Regression | 96.9% | 100.0% |
| Random Forest | 98.0% | 100.0% |

**Production choice: Logistic Regression** — comparable accuracy, faster inference, interpretable coefficients (audit-friendly for customs), simpler model lifecycle.

### Calibrated confidence → confidence-routed UX

Top-1 prediction confidence varies meaningfully:
- **>90%**: auto-fill the HS code with one-click confirm
- **60–90%**: present top-3 as a pick list
- **<60% or split top-2**: trigger a clarifying question (knitted/woven? gender?)

This converts model uncertainty into a UX that asks the consignor the right question instead of guessing.

### Error analysis insight

Dominant failure mode: knitted vs. woven blouses (`61061000` vs. `62063000`) when product descriptions don't mention construction. **This is a data problem, not a model problem.** The product fix is upstream — a single follow-up question when the construction term is missing — not a bigger model.

## How to reproduce

```bash
# Install dependencies
pip install -r requirements.txt

# Generate the data
python data/build_apparel_data.py
python data/build_rules_database.py

# Run the notebook
jupyter notebook notebooks/02_hs_code_recommender.ipynb
```

## Tech stack

Python 3.12, pandas, numpy, scikit-learn, matplotlib, seaborn, Jupyter

## What's next

- **Embedding-based retrieval** — replace TF-IDF with sentence-transformers; expected to improve recall on rare codes and unseen vocabulary (e.g., "kids" descriptions)
- **Hierarchical classification** — predict 4-digit heading first, then 8-digit subcode, to scale beyond 30 codes to the full 600+ apparel taxonomy
- **AI-assisted upload validation** — vision-LLM pipeline to validate consignee document uploads against the rules engine output
- **Continuous learning loop** — feed broker corrections back into training data and rules database
