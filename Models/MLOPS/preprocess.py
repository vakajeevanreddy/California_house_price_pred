import joblib
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import GRU, Dense, Dropout, Input
from keras.models import Sequential
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load the dataset
data = pd.read_csv(
    r"C:\Users\lenovo\OneDrive\Desktop\California_housing\Data\raw\california_housing.csv"
)

# 2. Implement Features
data["rooms_per_household"] = data["total_rooms"] / data["households"]
data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
data["population_per_household"] = data["population"] / data["households"]
data.replace([np.inf, -np.inf], np.nan, inplace=True)

X = data.drop("median_house_value", axis=1)
y = np.log1p(data["median_house_value"])

# FIX 1: Correct train_test_split unpacking order (x_train, x_test, y_train, y_test)
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Pipeline Preprocessing
categoric_cols = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()
numeric_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

# FIX 2: Correct tuple syntax and column parameters in ColumnTransformer
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
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore", sparse_output=False
                        ),
                    ),
                ]
            ),
            categoric_cols,
        ),
    ]
)

# FIX 3: Fit ONCE on train set, then TRANSFORM test set (prevents leakage)
x_train_prep = preprocessor.fit_transform(x_train)
x_test_prep = preprocessor.transform(x_test)

# 4. Save preprocessor artifact
joblib.dump(preprocessor, "preprocessor.joblib")
print("Preprocessing completed and preprocessor saved successfully!")

#2. Define Function with Arguments
def train_and_register_ann(x_train_prep, x_test_prep, y_train, y_test):
    with mlflow.start_run(run_name="ANN_Dense_Model") as run:
        # Build ANN Model using dynamic feature shape
        ann_model = Sequential(
            [
                Input(shape=(x_train_prep.shape[1],)),
                Dense(128, activation="relu"),
                Dropout(0.2),
                Dense(64, activation="relu"),
                Dropout(0.2),
                Dense(32, activation="relu"),
                Dense(1, activation="linear"),
            ]
        )

        ann_model.compile(
            optimizer="adam", loss="mean_squared_error", metrics=["mae"]
        )

        early_stop = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        # Train
        ann_model.fit(
            x_train_prep,
            y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1,
        )

        # Evaluation in Dollar Space
        y_pred_log = ann_model.predict(x_test_prep).flatten()
        y_test_dollars = np.expm1(y_test)
        y_pred_dollars = np.expm1(y_pred_log)

        r2 = r2_score(y_test_dollars, y_pred_dollars)
        rmse = np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars))
        mae = mean_absolute_error(y_test_dollars, y_pred_dollars)

        # Log Parameters
        mlflow.log_param("architecture", "ANN_Dense")
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("epochs", 50)
        mlflow.log_param("batch_size", 32)

        # Log Metrics
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("rmse_dollar", rmse)
        mlflow.log_metric("mae_dollar", mae)

        # Log & Register Model in MLflow
        mlflow.keras.log_model(
            model=ann_model,
            artifact_path="ann_housing_model",
            registered_model_name="California_Housing_ANN",
        )

        print(f"[ANN] Run Completed. R²: {r2:.4f} | RMSE: ${rmse:,.2f}")


# 3. Call Function Passing the Prepared Variables
if __name__ == "__main__":
    train_and_register_ann(x_train_prep, x_test_prep, y_train, y_test)