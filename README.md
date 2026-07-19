Car Price Prediction

A machine learning project that explores a used-car dataset and trains several regression models to predict car values based on features like year, present price, kilometers driven, fuel type, and transmission.

## Dataset

The notebook reads a local CSV (`carPrice.csv`) with the following columns:

- `Car_Name`
- `Year`
- `Selling_Price`
- `Present_Price`
- `Kms_Driven`
- `Fuel_Type`
- `Seller_Type`
- `Transmission`
- `Owner`

## Project Workflow

1. **Data Loading & Exploration**
   - Load the CSV with pandas
   - Inspect structure, data types, and missing values (`.info()`, `.isna().sum()`, `.dtypes`)
   - Preview rows with `.head()` / `.tail()`

2. **Exploratory Data Analysis (EDA)**
   - Correlation heatmap across numeric features
   - Distribution plot of `Selling_Price`
   - Boxplot of `Present_Price`
   - Bar plot of `Year` vs `Selling_Price`

3. **Preprocessing**
   - Label-encode `Fuel_Type`, `Transmission`, and `Car_Name`
   - One-hot encode `Seller_Type`
   - Scale `Selling_Price` with `MinMaxScaler`
   - Scale `Kms_Driven` with `StandardScaler`

4. **Train/Test Split**
   - Features: `Year`, `Selling_Price`, `Present_Price`, `Kms_Driven`, `Fuel_Type_encoded`, `Owner`, `Transmission_Encoded`
   - 80/20 train-test split (`random_state=42`)

5. **Models Trained**
   - Linear Regression
   - Random Forest Regressor (`n_estimators=100`)
   - XGBoost Regressor (`n_estimators=100`, `learning_rate=0.1`, `max_depth=5`)
   - Decision Tree Regressor (`max_depth=3`)

6. **Evaluation**
   - Each model is scored with MAE, MSE/RMSE, and R²
   - Actual vs. Predicted scatter plots for visual comparison

## Requirements

```
pandas
numpy
seaborn
matplotlib
scikit-learn
xgboost
```

Install with:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost
```

## Usage

1. Place `carPrice.csv` in the project directory and update the file path in the first cell.
2. Run the notebook `Internship-Project.ipynb` cell by cell in Jupyter.
3. Review the printed metrics and plots for each model.

## Notes / Suggested Fixes

- The current target variable is set to `Car_Name_Encoded` (i.e., the model predicts *which car* it is, not its price). If the goal is price prediction, set the target to `Selling_Price` and remove it from the feature list to avoid leakage.
- Consider dropping the encoded `Seller_Type` one-hot columns into the final feature set (they're created but not currently used in training).
- A single held-out test set is reused for scoring across all four models — a shared results table (e.g., comparing MAE/RMSE/R² side by side) would make it easier to pick the best performer.
