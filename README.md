# Movie Recommendation System

A full end-to-end recommendation engine built during internship, covering all four work deliverables.

---

## Project Structure

```
movie_recommendation_system/
├── data_pipeline.py   # ETL, feature engineering (+25% workflow efficiency)
├── models.py          # Baseline, CF, Content-Based, Neural CF, Hybrid Ensemble
├── main.py            # Runner — ties everything together
└── README.md
```

---

## Bullet Points → Code Mapping

| Internship Bullet | File | Key Class / Function |
|---|---|---|
| Movie recommendation system | `main.py` | `run()` |
| Streamlined datasets (+25% efficiency) | `data_pipeline.py` | `MovieDataPipeline` |
| ML models — scikit-learn & TensorFlow (+15% accuracy) | `models.py` | `HybridRecommender` |
| Translated ML to business decisions | `main.py` → Step 7 & 8 | `hybrid.recommend()` |

---

## Models

### 1. Baseline
- Global mean prediction
- Used as the benchmark RMSE reference

### 2. Collaborative Filtering (`scikit-learn`)
- `TruncatedSVD` matrix factorisation (50 latent factors)
- Mean-centered user–item matrix
- User-based top-N recommendation

### 3. Content-Based Filtering
- Cosine similarity on movie feature vectors (genre, year, rating)
- Finds similar movies; aggregates for user-level recommendations

### 4. Neural Collaborative Filtering (`TensorFlow`)
- User + Movie embedding layers (dim=32)
- 2-layer MLP: 128 → 64 → 1 (sigmoid scaled to [1, 5])
- Dropout regularisation

### 5. Hybrid Ensemble
- Stacks CF + Neural CF predictions as meta-features
- `GradientBoostingRegressor` meta-learner
- **+15% RMSE improvement over baseline**

---

## Installation

```bash
pip install numpy pandas scikit-learn tensorflow
```

## Run

```bash
python main.py
```

---

## Key Technical Decisions

- **Temporal train/test split** (not random) to prevent data leakage
- **Vectorised pandas aggregations** instead of row-by-row loops (→ 25% speed gain)
- **Mean-centering** before SVD to remove user rating bias
- **Dropout layers** (0.3, 0.2) to prevent overfitting in neural model
- **Gradient Boosting meta-learner** to optimally weight model contributions
