# Customer Segmentation: A Key to Unlocking Business Growth and Success

**Machine Learning II — NOVA IMS, 2024-2025**  
**Grade: 17/20**  
**Group 4:** Bernardo Caldas, David Santos, Inês Vicente

---

## Overview

Retail businesses generate vast amounts of customer data, but extracting actionable insights requires more than simple analysis. In this project, unsupervised machine learning techniques were applied to a real-world retail dataset to identify distinct customer segments and develop targeted marketing strategies tailored to each one.

**Result: 9 distinct customer profiles**, each with personalised promotional strategies backed by association rule mining using the Apriori algorithm.

---

## The Problem

Given two datasets — customer demographics and spending behaviour (`customer_info`) and transaction history (`customer_basket`) — the goal was to:

1. Identify meaningful customer segments based on purchasing patterns and demographics
2. Understand the motivations and behaviours driving each segment
3. Design targeted promotions and campaigns personalised to each segment

---

## Dataset

| Dataset | Description |
|---|---|
| `customer_info.csv` | Demographics, spending habits, loyalty card, complaints, location (Lisbon area) — 24 features |
| `customer_basket.csv` | Transaction history with product lists per customer |

All customers were located in the **Lisbon area**, confirmed via geographical mapping with Folium.

---

## Methodology

### 1. Exploratory Data Analysis
- Distribution analysis using count plots (discrete variables) and histograms (continuous variables)
- Key findings: right-skewed spending distributions, shopping peaks at 9h and 13h, most customers active since mid-2000s
- Geographical mapping of all customers across Lisbon using latitude/longitude

### 2. Data Preprocessing
Feature engineering:
- `customer_birthdate` → `customer_age`
- `year_first_transaction` → `years_since_first_transaction`
- `loyalty_card_number`: missing values → 1 (no card), non-missing → 0 (has card)
- `percentage_of_products_bought_promotion`: binarised (negative values → 0)
- `customer_gender`: encoded as binary (female=0, male=1)

Missing value treatment:
- KNN Imputation (k=5, distance-weighted) across 9 features
- Integer rounding applied post-imputation to discrete features

Outlier removal:
- **Unidimensional**: IQR-based removal of 19 high-extreme outliers across spending variables
- **Multidimensional**: DBSCAN (eps=2.55, min_samples=20) identified and removed 179 outliers

Scaling:
- **Robust Scaler** applied — chosen specifically for its robustness to remaining outliers

Feature selection:
- Correlation matrix analysis — no features removed (removing correlated features worsened clustering)
- `customer_gender` and `percentage_of_products_bought_promotion` dropped due to minimal variation across cluster profiles

### 3. Clustering

Multiple algorithms evaluated using **silhouette scores** and **UMAP visualisation**:

| Model | Notes |
|---|---|
| K-Means | Primary model — Elbow Method used for k selection |
| Hierarchical Clustering | Ward linkage |
| DBSCAN | Also used for multidimensional outlier removal |
| MiniSom (Self-Organizing Maps) | Grid search over hyperparameters |

After **5 iterations of refinement**, the final solution used **K-Means with 9 clusters** (silhouette score: 0.198), selected based on silhouette analysis, UMAP visualisation, and cluster interpretability.

### 4. Association Rules

The **Apriori algorithm** (mlxtend) was applied independently to each customer segment:
- Minimum support: 0.02
- Minimum confidence: 0.2
- Results sorted by lift to identify the most meaningful product combinations per cluster

---

## Customer Segments

| Cluster | Profile | Most Purchased Item | Key Characteristic |
|---|---|---|---|
| 1 | Vegan People | Tomatoes | High vegetable spend; strong preference for plant-based products |
| 2 | High Spenders | Champagne | Highest overall spend across all categories |
| 3 | Young Families with Issues | Oil | Concentrated near Cidade Universitária; basic product focus |
| 4 | Loyal Stable Shoppers | Oil | High loyalty card usage; consistent purchasing patterns |
| 5 | Tech-Gamer Customers | Energy Drinks | High electronics spend; energy drink + protein bar combinations |
| 6 | University Low-Spenders | Beer | Low overall spend; high alcohol consumption relative to budget |
| 7 | Hygiene Obsessed | White Wine | Disproportionately high hygiene product spend |
| 8 | Loyal Average Spenders | Oil | Stable moderate spend; loyal card holders |
| 9 | Big Family | Baby Food | Large household composition; multiple children |

---

## Marketing Strategies

Based on clustering and association rules, targeted promotions were designed per segment:

**Cross-segment:**
- Buy 1 Get 1 Free on Oil — applicable across multiple segments
- 15% Discount on All Purchases After Buying 30 Bottles of Oil
- Loyalty Card Birthday Gift (birthday cake)

**Segment-specific:**
- **Vegan People** — Monthly 20% discount on a rotating vegetable; 10% discount on mixed vegetable bundles
- **High Spenders** — Champagne gift for top monthly spenders
- **University Low-Spenders** — 30% student card discount on alcohol purchases over €20
- **Tech-Gamer Customers** — 15% discount on energy drinks + protein bars; 5% electronics discount per 25 energy drinks purchased
- **Big Family** — 10% discount on games after purchasing 20 baby foods; Ratchet & Clank bundle discount

---

## Tech Stack

- **Python** — pandas, NumPy
- **Scikit-Learn** — KMeans, AgglomerativeClustering, DBSCAN, RobustScaler, KNNImputer
- **MiniSom** — Self-Organizing Maps
- **mlxtend** — Apriori, association_rules, TransactionEncoder
- **UMAP** — dimensionality reduction for cluster validation
- **Folium** — geographical mapping of customers across Lisbon
- **Matplotlib / Seaborn** — data visualisation

---

## Repository Structure

```
customer-segmentation-ml/
│
├── README.md
├── src/
│   └── functions/
│       ├── __init__.py
│       └── preprocessing_function.py
├── notebooks/
│   ├── 1_data_analysis.ipynb
│   ├── 2_preprocessing.ipynb
│   ├── 3_models.ipynb
│   └── 4_association_rules.ipynb
├── data/
│   ├── data_customer_info.csv
│   └── data_customer_basket.csv
└── outputs/
    └── scored_customers.csv
```

---

## How to Run

Follow the notebooks in order:

1. `1_data_analysis.ipynb` — exploratory analysis and geographical visualisation
2. `2_preprocessing.ipynb` — feature engineering, imputation, outlier removal, scaling
3. `3_models.ipynb` — clustering models, evaluation, and final segment assignment
4. `4_association_rules.ipynb` — Apriori algorithm per segment and promotion design

The reusable preprocessing pipeline is available in `src/functions/preprocessing_function.py` and is imported directly into the modelling notebook.
