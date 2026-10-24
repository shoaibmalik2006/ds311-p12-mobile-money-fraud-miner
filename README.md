# P12 · Mobile-Money Fraud Miner

**DS-311 Data Mining · BS-AI 2024F, Section A · NBC, NUST · Fall 2026 · Group 2**

| Member | Role |
|---|---|
| Shoaib Malik (Team Leader) | Mining lead |
| Uzair Kakar | Data lead |
| Salman Hussian | Product lead |

**Live app:** _add your Streamlit link here_ · **Notebook:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/ds311-p12-mobile-money-fraud-miner/blob/main/notebooks/01_data_quality_eda.ipynb)

---

# Project Proposal (M0)

_Submitted for the Week 6 lab, 14 October 2026. A PDF copy is in [`docs/P12_Project_Proposal.pdf`](docs/P12_Project_Proposal.pdf)._

## 1. Problem Statement

Mobile wallets such as JazzCash and Easypaisa are growing fast in Pakistan, and fraudulent transfers hurt customers and providers. Fraud is extremely rare (about 0.1% of transactions), so a model that calls everything "normal" is 99.9% accurate yet catches nothing.

**We ask: how can a mobile-wallet service catch fraudulent transfers when fraud is extremely rare, explain each alert, and balance missed fraud against false alarms?**

## 2. Research Questions

| # | Research question | How we answer it |
|---|---|---|
| RQ1 | **Which catches what?** Do supervised models (Random Forest, XGBoost, LightGBM) and unsupervised outlier detectors (Isolation Forest, LOF, One-Class SVM) catch **different** frauds, and can a **hybrid score** combining both find frauds that each misses alone? | Same time-based test set for all models; overlap of caught frauds by transaction type and amount band; PR-AUC and recall at 90% precision |
| RQ2 | **Realism gap:** how much of the "near-perfect" performance seen in public notebooks comes from post-transaction balance columns, and how much remains when we use only information available at decision time? | Ablation of full vs "deployable" feature sets; leakage check on `isFlaggedFraud`; effect of training sampling rate on metrics at the natural 0.1% test rate |
| RQ3 | **Alert budget and cost:** if analysts can review only K alerts per day and a missed fraud costs far more than a false alarm, which model and threshold maximise value, and how sensitive is that choice to our PKR cost assumptions? | Calibrated probabilities, cost curves, precision and recall at K, sensitivity grid over the cost ratio |

## 3. Dataset at a Glance

| Attribute | Detail |
|---|---|
| **Source** | PaySim synthetic mobile-money transactions (Kaggle, `ealaxi/paysim1`), simulated from real mobile-money logs; loaded in Colab from Kaggle. Cite Lopez-Rojas et al., EMSS 2016; license as stated on Kaggle. |
| **Size and rarity** | About 6.3 million transactions over hourly steps (about 30 days); fraud about 0.1%. |
| **Types** | CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER (fraud expected only in TRANSFER and CASH-OUT; verified in EDA). |
| **Key columns** | `step` (hour), `type`, `amount`, sender and recipient balances before/after, `isFraud` (target), `isFlaggedFraud` (simulator rule flag). |
| **Known pitfalls** | Extreme imbalance; post-transaction balances can leak the answer; `isFlaggedFraud` is a rule output, so it is checked and not used as a feature. |
| **Handling plan** | Train on all frauds + 5 to 10% of normals; validation and test keep the natural fraud rate; time-based split by `step`; sampling effects reported. |
| **Why it fits** | Synthetic data avoids privacy problems, stays realistic, and is directly relevant to Pakistan's wallet growth. |

## 4. Models and Techniques

| Family | Models / techniques | Purpose |
|---|---|---|
| **Baselines** | Dummy classifier, Logistic Regression (class weights), Decision Tree | Show the accuracy trap; set the bar to beat |
| **Supervised ensembles** | Random Forest, Extra Trees, Balanced Random Forest, XGBoost, LightGBM | Strongest labelled-data detectors |
| **Unsupervised outliers** | Isolation Forest, LOF, One-Class SVM (on a sample), optional autoencoder-style reconstruction error | Find unusual transactions without labels |
| **Hybrid** | Weighted / stacked score of supervised probability + anomaly score | Catch what each misses (RQ1) |
| **Imbalance and calibration** | Class weights, undersampling, SMOTE inside training folds only, probability calibration (Platt / isotonic) | Honest probabilities for cost-based thresholds (RQ3) |
| **Explainability** | SHAP reasons, rule-based anomaly flags (account drained, destination balance zero, balance mismatch) | Reason for every alert |
| **Optional expert** | Account velocity features; TRANSFER to CASH-OUT chains as graph features | Bonus, only after M4 |

## 5. Tech Stack and Ideas Adopted from Reference Repositories

| Layer | Tools | Idea taken (inspiration only; our own code, disclosed) |
|---|---|---|
| **Data and ML** | Python, pandas, NumPy, SciPy, scikit-learn Pipelines, imbalanced-learn, joblib | Model comparison and imbalanced-learn use ([nehalverma355](https://github.com/nehalverma355-commits/fraud-detection)); tuning and feature importance ([fcamuz](https://github.com/fcamuz/fraud-detection-for-mobile-transactions)) |
| **Boosting and explainability** | XGBoost, LightGBM, SHAP | XGBoost with threshold and financial-impact analysis ([stevephilipgit](https://github.com/stevephilipgit/Fraud-Detection-Mobile-Financial-Transactions)) |
| **Risk logic** | Calibrated probabilities, optimized threshold, risk levels, five interpretable flags | [mobile-money-fraud-detector (Louizz7)](https://github.com/Louizz7/mobile-money-fraud-detector); account-draining pattern focus ([valentin396](https://github.com/valentin396/mobile-money-fraud-detection-)) |
| **Anomaly detection** | Isolation Forest, LOF, One-Class SVM | Unsupervised angle ([Lesson2003](https://github.com/Lesson2003/fraud_detection_app): Isolation Forest / autoencoder) |
| **App and delivery** | Streamlit + Plotly (analyst console, threshold slider), GitHub, Google Colab | Streamlit decision dashboard ([chetanapk27](https://github.com/chetanapk27/Data-Analytics-Project)); we keep one app instead of a separate Flask API |

## 6. Team Roles

| Member | Role | Techniques owned (viva) |
|---|---|---|
| **Shoaib Malik (Leader)** | Mining lead: evaluation, baselines, threshold and cost, integration | Decision Tree, Random Forest, Extra Trees, cost-based threshold |
| **Uzair Kakar** | Data lead: loading, quality report, EDA, features, leakage check | Isolation Forest, LOF, One-Class SVM, hybrid score |
| **Salman Hussian** | Product lead: GitHub, Streamlit app and deployment, report layout | XGBoost, LightGBM, calibration, SHAP |

## 7. Expected Outputs

- **Reproducible Colab notebook** (runs top-to-bottom, fixed seeds, scikit-learn Pipelines) and this **GitHub repository** with README, requirements and commits from all three members.
- **Public Streamlit app:** dashboard of fraud patterns, an analyst console (enter a transaction, get fraud probability, risk level and reasons) and a threshold slider showing the precision / recall / cost trade-off.
- **5 to 6 page report**, presentation with live demo, and individual vivas.

## 8. Milestones

| M0 · Wk 6 | M1 · Wk 7 | M2 · Wk 8 | M3 · Wk 11 | M4 · Wk 13 | Freeze · Wk 15 | Final · Wk 16-17 |
|---|---|---|---|---|---|---|
| 14 Oct<br>Proposal | 21 Oct<br>Data quality and EDA | 28 Oct<br>Transformations and first baseline | 18 Nov<br>Core complete, app skeleton | 2 Dec<br>Full evaluation, public app | 16 Dec<br>No new features, rehearsal | 21 Dec to 30 Dec<br>Submission, demo, viva |

## 9. Ethics and Disclosure

- **Privacy:** PaySim is synthetic; no personal or real customer data is collected or used.
- **Honest reporting:** sampling, the TRANSFER / CASH-OUT filter and the leakage checks are stated openly, and results on synthetic data may not transfer directly to real Pakistani wallets.
- **Assumptions:** all cost figures (fraud loss, review cost per alert in PKR) are our own labelled assumptions, not real provider data.
- **Impact of errors:** false alarms inconvenience genuine customers and missed fraud harms them, so thresholds are chosen by explicit cost, not by accuracy.
- **Attribution:** dataset, license and the PaySim paper are cited; reference repositories are used as inspiration only; any generative-AI help is disclosed in the report.

---

# How to Run

```bash
git clone https://github.com/<your-username>/ds311-p12-mobile-money-fraud-miner.git
cd ds311-p12-mobile-money-fraud-miner

# Analysis (or open the notebook in Colab using the badge above)
pip install -r requirements-notebook.txt

# Web app
pip install -r requirements.txt
streamlit run app/app.py
```

The dataset is **not** stored in this repo. The notebook downloads PaySim from Kaggle (credentials go in Colab Secrets, never in the repo).

# Repository Structure

```
notebooks/   Colab notebooks (data quality + EDA, features, models, evaluation)
src/         Reusable code (features, pipeline, metrics)
app/         Streamlit app
models/      Saved model pipelines (small files only)
docs/        Proposal PDF and report
CONTRIBUTIONS.md   Weekly log of who did what
```

# Citation and License

E. A. Lopez-Rojas, A. Elmir and S. Axelsson, "PaySim: A financial mobile money simulator for fraud detection", 28th European Modeling and Simulation Symposium (EMSS), 2016. PaySim dataset on Kaggle (`ealaxi/paysim1`); dataset license: _as stated on the Kaggle page (add here)_.

# AI-Use Disclosure

_Describe here any use of generative AI or reference code (what it was used for and what the team changed or verified)._
