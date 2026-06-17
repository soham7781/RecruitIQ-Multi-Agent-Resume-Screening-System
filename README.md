# Multi-Agent Resume Screening & Skill Mining System

A modular, multi-agent machine learning system for resume classification across **24 job categories**, built as part of CSE 572 (Data Mining) at Arizona State University.

---

## Overview

Traditional resume screening relies on keyword matching, which misses contextual meaning and doesn't scale. This system addresses that by combining three **specialized agents** — each capturing a different dimension of resume content — whose outputs are fused by a meta-classifier for a final prediction.

---

## System Architecture

```
Resume Text
    │
    ├──► Lexical Agent       (TF-IDF + LinearSVC)
    ├──► Semantic Agent      (Sentence-BERT + MLP)
    └──► Structured Skill Agent (XGBoost, 27 hand-crafted features)
                │
                ▼
         Fusion Agent (Weighted Average Meta-Classifier)
                │
                ▼
         Predicted Job Category (1 of 24)
```

### Agents

| Agent | Model | Key Strength |
|---|---|---|
| **Lexical** | TF-IDF (n-grams 1–2) + LinearSVC | Vocabulary-rich categories (FINANCE, AVIATION, DESIGNER) |
| **Semantic** | `all-MiniLM-L6-v2` SBERT + MLP | Semantically ambiguous roles (CONSULTANT, HR, SALES) |
| **Structured Skill** | XGBoost on 27 engineered features | Niche domain-identity categories (CHEF, PUBLIC-RELATIONS) |
| **Fusion** | Weighted Average (0.50 / 0.35 / 0.15) | Combines complementary signals across all 24 categories |

---

## Dataset

- **Source:** [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
- **Size:** 2,484 resumes across 24 job categories
- **Split:** 70% train / 15% validation / 15% test (stratified)
- **Features used:** `Resume_str` (plain text) and `Category` label

---

## Results

| Agent | Test Accuracy | Test Macro F1 |
|---|---|---|
| Lexical (TF-IDF + LinearSVC) | 0.7292 | 0.6721 |
| Semantic (SBERT + MLP) | 0.7051 | 0.6343 |
| Structured (XGBoost) | 0.4879 | 0.4550 |
| **Fusion (Weighted Average)** | **0.7239** | **0.6668** |

- The Lexical Agent achieved the strongest individual Macro F1 (0.6721).
- The Fusion Agent significantly outperforms the Structured Agent alone (paired t-test: *p* < 0.001).
- Agents exhibit **complementary failure patterns** — the key empirical motivation for fusion.

---

## Preprocessing Pipeline

Raw resume text was cleaned through the following steps:
1. Lowercasing
2. Removal of URLs, emails, and HTML tags
3. Removal of special characters (preserving alphanumeric content)
4. Stopword removal (NLTK English corpus)
5. Lemmatization (WordNet lemmatizer)

---

## Tech Stack

- **Language:** Python
- **ML Libraries:** scikit-learn, XGBoost, PyTorch
- **NLP:** NLTK, Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Evaluation:** Macro F1, Accuracy, Paired t-tests, Confusion matrices

---

## Team

Built by **Arjun Agrawal, Soham Kumar, Srijan Girdhar, and Kiran Kumar Sampath Kumar** at Arizona State University (Tempe, AZ) for CSE 572 — Data Mining.

---

## Future Work

- Cross-validation-based stacking for the meta-classifier
- Domain-adaptive fine-tuning of transformer embeddings on recruitment corpora
- NER-based skill extraction to strengthen the Structured Agent
- SMOTE or similar augmentation for minority categories (BPO, AUTOMOBILE)
