import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
def load_data(file_path):
    """Load the California Housing dataset from CSV."""
    df = pd.read_csv(file_path)
    return df
def handle_missing_values(df):
    """Handle missing values in the dataset."""
    # Check missing values
    print("\nMissing values before handling:")
    print(df.isnull().sum())
    # Fill missing total_bedrooms with mean
    if "total_bedrooms" in df.columns:
        df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].mean())
    print("\nMissing values after handling:")
    print(df.isnull().sum())
    return df
def check_duplicates(df):
    """Check and remove duplicate rows."""
    duplicate_count = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicate_count}")
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print("Duplicate rows removed.")
    return df
def preprocess_features(df):
    """Perform numerical scaling and categorical encoding."""
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    # Identify numerical columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    print("\nCategorical columns:")
    print(categorical_cols)
    print("\nNumerical columns:")
    print(numeric_cols)
    # Numerical preprocessing
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    # Categorical preprocessing
    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )
    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_cols
            ),
            (
                "cat",
                categorical_transformer,
                categorical_cols
            )
        ]
    )
    # Fit and transform
    data_preprocessed = preprocessor.fit_transform(df)
    # Get encoded categorical feature names
    cat_features = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_cols)
    # Combine feature names
    all_features = (numeric_cols+ list(cat_features) )
    # Convert processed data to DataFrame
    processed_df = pd.DataFrame(data_preprocessed,columns=all_features,index=df.index)
    return processed_df, preprocessor
# ---------------------------------------------------------
# 5. Complete Preprocessing Pipeline
# ---------------------------------------------------------
def preprocess_data(file_path):
    """Complete preprocessing pipeline."""
    # Load data
    df = load_data(file_path)
    print("\n========== FIRST 5 ROWS ==========")
    print(df.head())

    print("\n========== LAST 5 ROWS ==========")
    print(df.tail())

    print("\n========== DATASET INFORMATION ==========")
    df.info()

    # Missing values
    df = handle_missing_values(df)

    # Duplicates
    df = check_duplicates(df)

    # Preprocess features
    processed_df, preprocessor = preprocess_features(df)

    print("\n========== PROCESSED DATA ==========")
    print(processed_df.head())

    print("\nProcessed data shape:")
    print(processed_df.shape)

    return processed_df, preprocessor

# ---------------------------------------------------------
# 6. Main Program
# ---------------------------------------------------------

if __name__ == "__main__":

    file_path = (
        r"C:\Users\lenovo\OneDrive\Desktop"
        r"\California_housing\Data\raw"
        r"\california_housing.csv")
    processed_data, preprocessor = preprocess_data(file_path )
    print("\n========== PREPROCESSING COMPLETED ==========")
    print(processed_data.head())