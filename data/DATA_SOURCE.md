# Data Source Documentation

## Dataset Name
Crop Yield Prediction Dataset

## Source
- **Platform**: Kaggle
- **URL**: https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset
- **Version used**: 7
- **Downloaded with**: `kagglehub.dataset_download("patelris/crop-yield-prediction-dataset")`
- **File used**: `yield_df.csv`

## Description
Country-level yearly statistics on crop yield for various crop types, spanning
1990–2013 and covering ~100 countries worldwide.

## Target Variable
`hg/ha_yield` — crop yield in hectograms per hectare (hg/ha).
This is the continuous variable we aim to predict.

## Feature Variables
| Column | Type | Description |
|--------|------|-------------|
| `Area` | categorical | Country name |
| `Item` | categorical | Crop type (e.g. Wheat, Maize, Rice) |
| `Year` | integer | Year of observation (1990–2013) |
| `average_rain_fall_mm_per_year` | float | Mean annual rainfall (mm) |
| `pesticides_tonnes` | float | Pesticides used (tonnes) |
| `avg_temp` | float | Mean annual temperature (°C) |

## Why This Dataset is Suitable for Regularization Study
1. After one-hot encoding `Area` and `Item`, the feature space becomes large
   (several hundred columns), which encourages overfitting in OLS and gives
   regularization something meaningful to do.
2. Adding polynomial features (degree 2) further expands the feature space
   and makes the bias–variance trade-off visible.
3. The dataset is well-known, public, and reproducible via Kaggle.

## Limitations and Biases
1. **Country-level aggregation**: Both the target and all features are yearly
   country-level aggregates, not field-level measurements. This coarse
   granularity hides within-country variation and can mask real causal
   relationships.
2. **Non-independent rows**: Rows for the same country/crop combination across
   years are correlated. A standard random train/test split may leak
   information across time, making test performance optimistically high.
3. **Confounding variables**: Many drivers of crop yield (soil quality, crop
   variety, irrigation, fertilizer application, government policies) are not
   included. The variables present are proxies at best.
4. **No fertilizer column**: Despite some descriptions of the dataset
   mentioning fertilizer, `yield_df.csv` does NOT contain a fertilizer column.
   The available numeric predictors are rainfall, pesticides, and temperature.
5. **Class imbalance by crop type**: Some crops or countries appear far more
   often than others, which can bias model coefficients.
6. **Limited time range**: Only 1990–2013 is covered; modern trends (climate
   change, precision agriculture) are underrepresented.

## Preprocessing Applied
- Dropped `Unnamed: 0` (row index artifact).
- One-hot encoded `Area` and `Item` (drop_first=True to avoid multicollinearity).
- Applied StandardScaler inside the pipeline (after optional polynomial expansion).
- No rows were removed (0 missing values, 0 exact duplicates in raw file).
