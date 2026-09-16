import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv(
    r"C:\Users\lenovo\OneDrive\Desktop\California_housing\Data\raw\california_housing.csv"
)

# Fill missing values
data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].mean())

# Separate features and target
X = data.drop('median_house_value', axis=1)
y = data['median_house_value']

# Identify numeric and categorical columns
categorical_cols = X.select_dtypes(include=['object', 'string']).columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessor: scale numeric, one-hot encode categorical
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Pipeline with LinearSVR
svr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearSVR(random_state=42, max_iter=10000))
])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'regressor__C': [1, 10, 100, 1000],
    'regressor__epsilon': [0.01, 0.1, 1, 10]
}

grid_search = GridSearchCV(
    svr_pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1
)

# Fit GridSearchCV on training set
grid_search.fit(x_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best CV R²:", grid_search.best_score_)

# Retrain on full training set with best params
best_svr = grid_search.best_estimator_
best_svr.fit(x_train, y_train)

# Predictions
y_pred = best_svr.predict(x_test)

# Evaluations
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)
