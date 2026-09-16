import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load dataset
data = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\California_housing\Data\raw\california_housing.csv")

# Fill missing values
data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].mean())

X = data.drop('median_house_value', axis=1)
Y = data['median_house_value']

# Identify numeric and categorical columns
categorical_cols = X.select_dtypes(include=['object', 'string']).columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Pipeline with DecisionTreeRegressor
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42))
])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Correct parameter grid
dt_params = {
    'regressor__max_depth': [None, 5, 10, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4, 7, 10]
}

# GridSearchCV
grid_search_dt = GridSearchCV(dt_pipeline, dt_params, cv=3, scoring='r2', n_jobs=-1)
grid_search_dt.fit(x_train, y_train)

# Predictions
dt_pred = grid_search_dt.predict(x_test)

# Evaluation
print("Mean Absolute Error:", mean_absolute_error(y_test, dt_pred))
print("Mean Squared Error:", mean_squared_error(y_test, dt_pred))
print("R² Score:", r2_score(y_test, dt_pred))
