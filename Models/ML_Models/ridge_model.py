import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Load dataset
data = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\California_housing\Data\raw\california_housing.csv")

# Fill missing values
data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].mean())

# Separate features and target
X = data.drop('median_house_value', axis=1)
Y = data['median_house_value']

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'string']).columns

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Pipeline with Ridge
ridge_pipeline = Pipeline(steps=[('preprocessor', preprocessor),('regressor', Ridge(max_iter=500, random_state=42))])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Ridge
param_grid = {
    'regressor__alpha': [0.01, 0.1, 1, 10, 100],
    'regressor__max_iter': [500, 1000, 2000]
}

grid_search_ridge = GridSearchCV(ridge_pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search_ridge.fit(x_train, y_train)

# Predictions
ridge_pred = grid_search_ridge.predict(x_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, ridge_pred))
print("Mean Squared Error:", mean_squared_error(y_test, ridge_pred))
print("R² Score:", r2_score(y_test, ridge_pred))
