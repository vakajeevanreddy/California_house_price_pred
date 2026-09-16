import os
import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# 1. Set MLflow Experiment
mlflow.set_experiment("California_Housing_XGBoost")

# 2. Load Dataset
data = pd.read_csv("Data/raw/california_housing.csv")

# 3. Feature Engineering
data["rooms_per_household"] = data["total_rooms"] / data["households"]
data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
data["population_per_household"] = data["population"] / data["households"]
data.replace([np.inf, -np.inf], np.nan, inplace=True)

X = data.drop("median_house_value", axis=1)
y_raw = data["median_house_value"]
y = np.log1p(y_raw)

# 4. Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Preprocessing Pipeline
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_cols,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore", sparse_output=False
                        ),
                    ),
                ]
            ),
            categorical_cols,
        ),
    ]
)

# 6. Fit Preprocessor & Transform Features
x_train_prep = preprocessor.fit_transform(x_train)
x_test_prep = preprocessor.transform(x_test)

# Save Preprocessor Artifact
joblib.dump(preprocessor, "preprocessor.joblib")

# 7. Model Hyperparameters & Training
xgb_params = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

xgb_reg = XGBRegressor(**xgb_params)
xgb_reg.fit(
    x_train_prep,
    y_train,
    eval_set=[(x_test_prep, y_test)],
    verbose=False,
)

# Save Local Fallback Model Artifact for Docker
os.makedirs("Models", exist_ok=True)
joblib.dump(xgb_reg, "Models/xgboost_model.joblib")

# 8. MLflow Logging
model_name = "California_Housing_XGBoost"

with mlflow.start_run(run_name="XGBoost_Production_Candidate"):
    y_pred_log = xgb_reg.predict(x_test_prep)
    y_test_dollar = np.expm1(y_test)
    y_pred_dollar = np.expm1(y_pred_log)

    r2 = r2_score(y_test_dollar, y_pred_dollar)
    rmse = np.sqrt(mean_squared_error(y_test_dollar, y_pred_dollar))
    mae = mean_absolute_error(y_test_dollar, y_pred_dollar)

    mlflow.log_params(xgb_params)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("rmse_dollar", rmse)
    mlflow.log_metric("mae_dollar", mae)

    signature = infer_signature(x_train_prep[:5], xgb_reg.predict(x_train_prep[:5]))
    mlflow.xgboost.log_model(
        xgb_model=xgb_reg,
        name="xgboost_housing_model",
        signature=signature,
        registered_model_name=model_name,
    )

try:
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=1,
        stage="Production",
        archive_existing_versions=True,
    )
except Exception as e:
    print(f"MLflow Stage Transition Warning: {e}")