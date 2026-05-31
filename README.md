# Indian Diseases Dataset — EDA & Preprocessing

End-to-end exploratory data analysis and preprocessing pipeline on a real-world Indian patient dataset of 20,000 records across 32 diseases. The processed dataset is ML-ready for XGBoost classification.

---

## Project Goal

EDA and feature engineering.

---

## Dataset

- **Source:** [Indian Patient Disease and Treatment Dataset](https://www.kaggle.com/datasets/ashyou09/indian-patient-disease-and-treatment-dataset) by ashyou09, licensed under **MIT**
- **Rows:** 20,000 patients
- **Diseases covered:** 32
- **Time period:** 2018–2025
- **Original columns:** 32 → **Final features (X):** 63 columns

> The raw dataset is not included in this repository.  
> It is downloaded automatically via `kagglehub` when you run the notebook.

---

## Repository Structure

```
├── app.ipynb                # Main EDA + preprocessing notebook)
├── README.md
└── .gitignore
```

---

## Key EDA Findings

### Target Distribution
| Outcome | Count | % |
|---|---|---|
| Recovered | 8,440 | 42.2% |
| Recovering | 5,253 | 26.3% |
| Chronic Management | 4,485 | 22.4% |
| Deceased | 1,822 | 9.1% |

Overall death rate: **9.11%** — imbalanced dataset, keep in mind for XGBoost class weights.

### Disease Mortality (Top 5 Deadliest)
| Disease | Death Rate |
|---|---|
| Stroke | 31.69% |
| Heart Disease | 25.08% |
| Lung Cancer | 23.27% |
| Oral Cancer | 22.70% |
| Chronic Kidney Disease | 22.38% |

Lowest mortality: Jaundice/Hepatitis A (0.47%), Gastroenteritis (0.76%), Diarrhea (0.79%).

### Severity vs Death Rate
| Severity | Death Rate |
|---|---|
| Mild | 2.37% |
| Moderate | 2.59% |
| Severe | 9.25% |
| Critical | 41.00% |

Severity is the **strongest single predictor** of death (correlation: 0.33). Critical patients die at 17× the rate of Mild patients.

### Age by Outcome
| Outcome | Mean Age |
|---|---|
| Deceased | 58.7 years |
| Recovering | 45.5 years |
| Chronic Management | 44.0 years |
| Recovered | 40.1 years |

Deceased patients are on average **18.6 years older** than recovered patients.

### Seasonal Patterns
- **Vector-borne diseases** (Malaria, Dengue, Chikungunya) peak heavily in **Monsoon** (1,629 cases vs 648 in Winter)
- **Highest death rate months:** March (11.59%), February (11.32%)
- **Lowest death rate months:** August (7.13%), July (7.59%)
- Winter has the highest seasonal death rate (10.08%) — driven by cardiac and respiratory diseases

### Geographic Patterns (Death Rate)
- **Highest:** Jammu & Kashmir (11.32%), Chhattisgarh (10.86%), Assam (10.58%)
- **Lowest:** Mizoram (0.00%), Puducherry (5.56%), Andaman & Nicobar (5.88%)

### Top Symptoms by Frequency
1. Fatigue (5,639 patients)
2. Nausea (4,148)
3. Fever (3,829)
4. Shortness of breath (3,365)
5. Chest pain (3,254)

### Top Features Correlated with Death
1. Severity (0.33)
2. Age (0.18)
3. Unexplained weight loss (0.16)
4. Dizziness (0.16)
5. Numbness (0.15)

---

## Preprocessing Steps

### Missing Value Handling
| Column | Missing | Strategy |
|---|---|---|
| `cause_of_death` | 18,178 (91%) | Dropped — no new information |
| `alcohol_use` | 6,627 (33%) | Left as NaN — XGBoost handles natively |
| `recovery_days` | 6,307 | Deceased → `max * 2` sentinel; Chronic → NaN |
| `comorbidity` | varies | Filled with `"No comorbidity"` |

### Encoding
| Column | Method | Reason |
|---|---|---|
| `severity` | OrdinalEncoder: Mild→0 … Critical→3 | Real medical order |
| `income_level` | OrdinalEncoder: BPL→0 … High→4 | Real socioeconomic order |
| `smoking_status` | OrdinalEncoder: Never→0, Former→1, Current→2 | Real risk order |
| `alcohol_use` | map(): Occasional→1, Regular→2, Heavy→3 | Real consumption order, preserves NaN |
| `season` | map(): Summer→1, Monsoon→2, Post-Monsoon→3, Winter→4 | Seasonal progression |
| `gender`, `state`, `disease_name`, etc. | LabelEncoder | Nominal — no meaningful order |
| `symptoms` | MultiLabelBinarizer → 44 binary columns | Comma-separated fixed vocabulary of 44 symptoms |

### Columns Dropped
| Column | Reason |
|---|---|
| `patient_id` | Unique identifier — zero predictive value |
| `age_group` | Duplicate of `age` |
| `city` | Redundant — `state` + `region` capture location |
| `diagnosis_date` | Raw date — `year`, `month`, `season` already extracted |
| `cause_of_death` | 91% missing, information already in `disease_name` |
| `treatment_type` | Post-admission leakage |
| `recovery_days` | Post-outcome leakage |
| `days_hospitalized` | Post-outcome leakage |
| `follow_up_required` | Post-outcome leakage |
| `treatment_cost_inr` | Post-admission leakage |

---

## Final Dataset

```
X.csv          →  20,000 rows × 63 columns  (admission features only)
y.csv          →  20,000 rows × 2 columns   (death_flag, outcome)
processed_df.csv → 20,000 rows × 65 columns (X + y combined)
```

Only `alcohol_use` retains NaN (6,627 rows) — intentional, XGBoost handles it natively via its default direction mechanism.

---

## Future Work

- [ ] Train XGBoost classifier on `death_flag` (binary classification)
- [ ] Train XGBoost classifier on `outcome` (4-class classification)
- [ ] Handle class imbalance (9.11% death rate) with `scale_pos_weight`
- [ ] Feature importance analysis
- [ ] Hyperparameter tuning
- [ ] Cyclical encoding for `season` and `month` (sine/cosine)
- [ ] Proper train/test split and cross-validation

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
missingno
kagglehub
xgboost
jupyter
```

Install with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn missingno kagglehub xgboost jupyter
```

---

## How to Run

1. Clone the repository
2. Set up Kaggle API credentials (`kaggle.json`) in `~/.kaggle/`
3. Open `app.ipynb` in Jupyter
4. Run all cells top to bottom
5. `X.csv`, `y.csv`, and `processed_df.csv` will be generated automatically


>> I am just a beginner stepping into field of Artificial Intelligence and Machine Learning. This project might have some mistakes as I am a beginner but with experience I will fix them all.
