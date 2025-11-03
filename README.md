# 📘 Project Overview

This project investigates whether machine learning algorithms can predict if a **bike-share user** is a **member** or a **casual rider**, based on trip metadata, station information, and weather conditions.

Understanding and predicting subscription types helps bike-sharing companies:

- 🚲 Optimize bike availability and fleet management  
- 💰 Tailor membership and pricing strategies  
- ⚙️ Improve operational efficiency and customer experience  

---

## 🧩 Dataset Description

| Dataset             | Source | Size                     | Key Features                                       |
|---------------------|--------|--------------------------|---------------------------------------------------|
| Daily Bike Rentals  | Kaggle | 1.6 GB (16,086,672 × 13) | Trip details (start time, station, duration, etc.) |
| Weather Data        | Kaggle | 408.5 KB (1,584 × 33)    | Temperature, humidity, wind, conditions, etc.      |

**Period Covered:** January 2023  
**Target Variable:** `member_casual` → `'member'` or `'casual'`

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing

- Removed missing / duplicate / unrealistic entries  
- Reduced high cardinality of stations by grouping into regions:  
  `northwest`, `northeast`, `southwest`, `southeast`, `metro_area`, `education`, `recreation`, `commercial`, `other`  
- Simplified weather text into categories:  
  `rainy`, `clear`, `cloudy`, `partly_cloudy`, `snowy`, `other`, `unknown`  
- Extracted temporal features:  
  `day_of_week`, `is_weekend`, `week_of_month`, `is_new_year_week`, `weekday_category`  
- Encoded categorical features with **OneHotEncoder**  
- Standardized numeric features with **StandardScaler**

---

### 2️⃣ Exploratory Data Analysis (EDA)

- Analyzed distributions and correlations  
- Applied **SelectKBest** for feature selection  
- Visualized **target imbalance** and **categorical cardinality**

---

### 3️⃣ Modelling Pipeline

- Stratified split: **80% training**, **20% validation**
- Models implemented:
  - Gaussian Naive Bayes  
  - k-Nearest Neighbors (default & tuned)  
  - Decision Tree (default & tuned)  
  - Random Forest (default & tuned)  
  - Voting Classifiers (equal / weighted / best models)
- Hyperparameter tuning: **GridSearchCV (cv=5)**
- Metrics used: **Accuracy** and **F1 Score**

---

## 🧠 Key Insights

- **Best Performing Models:** Tuned Random Forest & Voting Classifier (Best) — stable generalization across folds  
- **Top Predictors:** `rideable_type_classic_bike` and `rideable_type_electric_bike`  
- **Moderate Influence:** `station_region`  
- **Low Impact:** Weather-related features  
- **Overfitting Observed In:** KNN & Naive Bayes (default and tuned)  
- **Practical Value:** Supports strategic bike allocation and membership campaigns  

---

## 📈 Results Summary

| Model                  | Validation F1 | CV F1 Mean | Metric Gap |
|-------------------------|--------------:|------------:|------------:|
| Gaussian Naive Bayes    | 0.8231        | 0.5542      | 0.2689      |
| K-NN (default)          | 0.8124        | 0.5376      | 0.2748      |
| K-NN (tuned)            | 0.7844        | 0.5554      | 0.2289      |
| Decision Tree (default) | 0.7326        | 0.5883      | 0.1443      |
| Decision Tree (tuned)   | 0.7326        | 0.5883      | 0.1443      |
| Random Forest (default) | 0.7312        | 0.5887      | 0.1424      |
| Random Forest (tuned)   | 0.7304        | 0.5904      | 0.1400      |
| Voting (equal weights)  | 0.8326        | 0.5442      | 0.2884      |
| Voting (weighted)       | 0.8298        | 0.5537      | 0.2761      |
| Voting (best)           | 0.7302        | 0.5937      | 0.1365      |

---

## 📊 Visual Outputs

- 🔥 Correlation heatmaps  
- 🌲 Feature importance plots (Decision Tree & Random Forest)  
- 🧮 Confusion matrices  
- 📉 Model performance comparison charts  
- 🏆 Final model selection visuals  

---

## 🧪 Statistical Testing

Paired **t-tests** were conducted to compare top models (Best KNN vs Decision Tree vs Random Forest).  
✅ Results show that **Random Forest** and **Voting ensembles** significantly outperform simpler models (**p < 0.05**).

---

## 🧰 Technologies Used

| Category     | Tools / Libraries |
|---------------|------------------|
| Language      | Python 3.10+ |
| Core Libraries | pandas, numpy, matplotlib, seaborn, scipy |
| Machine Learning | scikit-learn (`NaiveBayes`, `KNN`, `DecisionTree`, `RandomForest`, `VotingClassifier`, `GridSearchCV`) |
| Environment   | Jupyter Notebook / Python Script (`.py`) |
| Visualization | Matplotlib & Seaborn (300 DPI plots) |

---

## ⚠️ Limitations & Future Work

### Current Limitations
- Temporal bias due to random 80/20 split (potential seasonal leakage)  
- Simplified weather categories ignore detailed meteorological effects  
- Regional grouping reduces fine-grained station-level insights  
- Cross-validation increases computational cost  

### Future Work
- Add **month** as a feature  
- Evaluate **multi-month or multi-year** data  
- Apply **permutation importance** for feature explainability  
- Benchmark results using a **hold-out test set**

---

## 🏁 Conclusion

Machine-learning models — particularly **Random Forest** and **Voting Classifiers** — can effectively distinguish **member vs casual riders** with **stable F1 scores** and **low variance**.  

This demonstrates the **feasibility of data-driven optimization** for modern **bike-share systems**, improving both operational and strategic decision-making.

---
