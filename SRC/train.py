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

# 2. Wrap the whole execution in a single MLflow context manager
with mlflow.start_run(run_name="XGBoost_Production_Candidate") as run:

    # Load Dataset using Relative Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    DATA_PATH = os.path.join(PROJECT_ROOT, "Data", "raw", "california_housing.csv")

    data = pd.read_csv(DATA_PATH)

    # 3. Feature Engineering
    data["rooms_per_household"] = data["total_rooms"] / data["households"]
    data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
    data["population_per_household"] = data["population"] / data["households"]
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Separate features and target
    X = data.drop("median_house_value", axis=1)
    y_raw = data["median_house_value"]
    y = np.log1p(y_raw)  # Log-transform target for stabilized training variance

    # 4. Train-Test Split (80/20 ratio)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training data shape:", x_train.shape, y_train.shape)
    print("Testing data shape:", x_test.shape, y_test.shape)

    # 5. Build Preprocessing Pipeline
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

    # Save Preprocessor Artifact locally
    joblib.dump(preprocessor, "preprocessor.joblib")
    print("[Artifact] Saved preprocessor.joblib successfully.")

    # 7. Model Hyperparameters
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

    # 8. Train XGBoost Model
    xgb_reg = XGBRegressor(**xgb_params)
    xgb_reg.fit(
        x_train_prep,
        y_train,
        eval_set=[(x_test_prep, y_test)],
        verbose=False,
    )

    # Save local model artifact as fallback
    os.makedirs("Models", exist_ok=True)
    joblib.dump(xgb_reg, "Models/xgboost_model.joblib")
    print("[Artifact] Saved Models/xgboost_model.joblib successfully.")

    # 9. MLflow Tracking & Metrics Logging
    model_name = "California_Housing_XGBoost"

    # Evaluation metrics calculated in original Dollar Space
    y_pred_log = xgb_reg.predict(x_test_prep)
    y_test_dollar = np.expm1(y_test)
    y_pred_dollar = np.expm1(y_pred_log)

    r2 = r2_score(y_test_dollar, y_pred_dollar)
    rmse = np.sqrt(mean_squared_error(y_test_dollar, y_pred_dollar))
    mae = mean_absolute_error(y_test_dollar, y_pred_dollar)

    # Log metrics & parameters
    mlflow.log_params(xgb_params)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("rmse_dollar", rmse)
    mlflow.log_metric("mae_dollar", mae)

    # Save model locally and upload as a pure file artifact (completely avoids /logged-models)
    local_model_path = "Models/xgboost_model.joblib"
    os.makedirs("Models", exist_ok=True)
    joblib.dump(xgb_reg, local_model_path)
    
    # Upload local file as an artifact
    mlflow.log_artifact(local_model_path, artifact_path="model_artifacts")

    print("\n=== XGBoost Model Evaluation ===")
    print(f"R² (Dollar Space):   {r2:.4f}")
    print(f"RMSE (Dollar Space): ${rmse:,.2f}")
    print(f"MAE (Dollar Space):  ${mae:,.2f}")

# 10. Transition to Production Stage
try:
    client = MlflowClient()
    latest_versions = client.get_latest_versions(name=model_name, stages=["None"])
    if latest_versions:
        latest_version = latest_versions[0].version
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"\n[MLflow] Successfully transitioned '{model_name}' Version {latest_version} to 'Production' stage!")
except Exception as e:
    print(f"\n[MLflow Warning] Stage transition failed: {e}")