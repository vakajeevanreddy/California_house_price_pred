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


def train_and_register_rnn():
    # Reshape features for Recurrent layers (samples, time_steps, features)
    x_train_rnn = np.reshape(
        x_train_prep, (x_train_prep.shape[0], 1, x_train_prep.shape[1])
    )
    x_test_rnn = np.reshape(
        x_test_prep, (x_test_prep.shape[0], 1, x_test_prep.shape[1])
    )

    with mlflow.start_run(run_name="RNN_GRU_Model") as run:
        # Build RNN Model
        rnn_model = Sequential(
            [
                Input(shape=(1, x_train_prep.shape[1])),
                GRU(64, return_sequences=True),
                Dropout(0.2),
                GRU(32),
                Dense(16, activation="relu"),
                Dense(1, activation="linear"),
            ]
        )

        rnn_model.compile(
            optimizer="adam", loss="mean_squared_error", metrics=["mae"]
        )

        early_stop = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        # Train
        history = rnn_model.fit(
            x_train_rnn,
            y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0,
        )

        # Evaluation in Dollar Space
        y_pred_log = rnn_model.predict(x_test_rnn).flatten()
        y_test_dollars = np.expm1(y_test)
        y_pred_dollars = np.expm1(y_pred_log)

        r2 = r2_score(y_test_dollars, y_pred_dollars)
        rmse = np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars))
        mae = mean_absolute_error(y_test_dollars, y_pred_dollars)

        # Log Parameters
        mlflow.log_param("architecture", "RNN_GRU")
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("epochs", 50)
        mlflow.log_param("batch_size", 32)

        # Log Metrics
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("rmse_dollar", rmse)
        mlflow.log_metric("mae_dollar", mae)

        # Log and Register Model in MLflow Registry
        mlflow.keras.log_model(
            model=rnn_model,
            name="rnn_housing_model",
            registered_model_name="California_Housing_RNN",
        )

        print(f"[RNN] Run Completed. R²: {r2:.4f} | RMSE: ${rmse:,.2f}")


train_and_register_rnn()