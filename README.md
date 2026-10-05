# Student Performance Prediction (Final Grade G3)

**Synent Technologies – Data Science Internship · Task 8 (Machine Learning Model)**

## Problem statement

Schools want to flag students at risk of a low final grade early enough to intervene. This project predicts the **final grade G3 (0–20 scale)** of secondary-school students from background, school-life and prior-grade features, and compares three regressors (Linear Regression, Random Forest, HistGradientBoosting) using RMSE, MAE and R².

Task 8 requirements covered: data preprocessing, feature selection, train/test split, model training (Linear Regression, Random Forest), RMSE evaluation — plus MAE, R², hyperparameter tuning and a deployed Streamlit demo.

## Dataset details

- **Source:** Student Performance, UCI Machine Learning Repository ([link](https://archive.ics.uci.edu/dataset/320/student-performance)) — P. Cortez & A. Silva, secondary schools in Portugal
- **Size:** 649 records, 33 columns, no missing values, no duplicates; target mean 11.91 (std 3.23)
- **Target:** `G3` final grade (0–20); `G1`/`G2` are first/second-period grades
- **Features (32):** demographics (age, sex, address, family), parents' education/jobs, study habits (studytime, failures, absences, extra support/classes), social life and health indicators + G1/G2
- The file `StudentPerformance.csv` is downloaded automatically (notebook cell / `download_dataset.py`) if missing

## Approach

1. **Data cleaning** – verified no missing values/duplicates, standardised target handling (predictions clipped to 0–20)
2. **EDA & visualisation** – G3 distribution, G3 vs G2, grades by study time and past failures, correlation matrix
3. **Feature selection** – all 30 background/school-life features kept plus G1/G2; a robustness test (section 13) quantifies performance without prior grades
4. **Train/test split** – random 80/20 (519 vs 130 rows, `random_state=42`)
5. **Modeling** – scikit-learn pipelines (median imputation + scaling / one-hot): Linear Regression, Random Forest, HistGradientBoosting
6. **Validation & tuning** – `KFold(5)` cross-validation, `RandomizedSearchCV` on the train set only, final evaluation on unseen students
7. **Deployment** – best model saved (`student_model_v1.pkl`) and served by `app.py` (Streamlit: user input → predicted grade with 80% interval)

## Results

Test set — students never seen in training:

| Model | RMSE | MAE | R² | R² train |
|---|---|---|---|---|
| Linear Regression | 1.215 | 0.765 | 0.849 | 0.858 |
| **Random Forest (selected)** | **1.204** | **0.727** | **0.851** | **0.961** |
| HistGradientBoosting | 1.266 | 0.738 | 0.836 | 0.962 |

- Model selection used cross-validation RMSE (RF 1.287 vs Linear 1.332 vs HGB 1.390), not the test set
- Top drivers (permutation importance, RMSE increase): G2 (+2.34), G1 (+0.27), absences (+0.09), age, past failures
- Honesty check without prior grades (G1/G2 removed): R² falls to ≈ 0.16–0.23 — background data alone carries little signal, which is exactly the realistic early-warning scenario

## Project structure

```text
├── Task8_StudentPerformance_ML.ipynb  # full workflow: cleaning → EDA → modeling → evaluation
├── app.py                              # Streamlit demo (input → predicted grade)
├── download_dataset.py                 # downloads the UCI dataset as StudentPerformance.csv
├── StudentPerformance.csv              # dataset (UCI, 649 rows) — also auto-downloaded if missing
├── student_model_v1.pkl                # deployed Random Forest + metadata (used by app.py)
├── student_model_v2.pkl                # variant with business rule + calibrated interval
├── requirements.txt
├── run_notebook.bat                    # install deps + open Jupyter (Windows)
└── run_app.bat                         # launch the Streamlit demo (Windows)
```

## Demo Video & Links

- **GitHub:** [synent-task8-studentperformance-ahmedaminebejaoui](https://github.com/AhmedAmineBejaoui/synent-task8-studentperformance-ahmedaminebejaoui)
- **Dataset:** [UCI — Student Performance](https://archive.ics.uci.edu/dataset/320/student-performance) + local `StudentPerformance.csv`
- **Video (1–3 min):** [Google Drive — Task 8 Demo](https://drive.google.com/file/d/1ETnTnrSMkgQ-t-QQUwcFpV2V9K6ewL7A/view?usp=sharing)
- **LinkedIn post:** [LinkedIn — Student Performance post](https://lnkd.in/p/em4FXMGD)

## How to run

```bash
python -m pip install -r requirements.txt
python download_dataset.py        # optional: the notebook also auto-downloads
jupyter notebook                  # open Task8_StudentPerformance_ML.ipynb, run top to bottom
streamlit run app.py              # launch the demo
```

## Author

Ahmed Amine Bejaoui — Synent Technologies Data Science Internship Program.
