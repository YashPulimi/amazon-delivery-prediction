# Amazon Last-Mile Delivery — ETA Prediction & Late Delivery Detection

End-to-end data science project on 43,739 real Amazon deliveries.

## Models
- **Delivery Time Regression** (LightGBM) — predicts actual delivery time using distance, weather, traffic, agent rating, and temporal features
- **Late Delivery Classifier** (LightGBM) — flags orders at risk of being late before dispatch

## Results
| Model | Metric | Score |
|-------|--------|-------|
| ETA Regression | MAE | 17.28 min |
| ETA Regression | RMSE | 22.09 min |
| Late Delivery Classifier | ROC-AUC | 0.964 |
| Late Delivery Classifier | Recall | 0.904 |
| Late Delivery Classifier | 5-Fold CV AUC | 0.957 ± 0.002 |

## Key Findings
- Traffic density and weather are the strongest delivery delay drivers (ANOVA p < 0.001)
- Agent rating has a significant negative correlation with delivery time (r = -0.260, p < 0.001)
- SHAP feature importance used to communicate top exception risk factors to non-technical stakeholders

## Dataset
[Amazon Delivery Dataset](https://www.kaggle.com/datasets/sujalsuthar/amazon-delivery-dataset) — Kaggle

## Tech Stack
Python, LightGBM, pandas, scikit-learn, SHAP, NumPy, SciPy, Matplotlib, ipywidgets
