import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "california_housing.csv"
PLOT_DIR = PROJECT_ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(file_path):
    print("\nLoading dataset...")
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    return df


def preprocess_data(df):
    df = df.copy()
    if "total_bedrooms" in df.columns:
        df["total_bedrooms"] = df["total_bedrooms"].fillna(
            df["total_bedrooms"].mean()
        )
    return df


class Visualization:

    def __init__(self, data, plot_dir):
        self.data = data
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def plot_histogram(self, column, filename=None):
        if filename is None:
            filename = f"{column}_histogram.png"
        plt.figure(figsize=(10, 6))
        plt.hist(
            self.data[column].dropna(),
            bins=50,
            edgecolor="black"
        )
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {column}")
        plt.tight_layout()
        save_path = self.plot_dir / filename
        plt.savefig(save_path)
        plt.show()
        plt.close()
        print(f"Histogram saved: {save_path}")

    def plot_correlation_heatmap(self, filename="correlation_heatmap.png"):
        numeric_data = self.data.select_dtypes(include=["int64", "float64"])
        correlation_matrix = numeric_data.corr()
        plt.figure(figsize=(12, 9))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        save_path = self.plot_dir / filename
        plt.savefig(save_path)
        plt.show()
        plt.close()
        print(f"Correlation heatmap saved: {save_path}")

    def plot_scatter(self, x_column, y_column, filename=None):
        if filename is None:
            filename = f"{x_column}_vs_{y_column}_scatter.png"
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data[x_column], self.data[y_column], alpha=0.5)
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f"{x_column} vs {y_column}")
        plt.tight_layout()
        save_path = self.plot_dir / filename
        plt.savefig(save_path)
        plt.show()
        plt.close()
        print(f"Scatter plot saved: {save_path}")

    def plot_box(self, column, filename=None):
        if filename is None:
            filename = f"{column}_boxplot.png"
        plt.figure(figsize=(8, 6))
        plt.boxplot(self.data[column].dropna())
        plt.ylabel(column)
        plt.title(f"Boxplot of {column}")
        plt.tight_layout()
        save_path = self.plot_dir / filename
        plt.savefig(save_path)
        plt.show()
        plt.close()
        print(f"Boxplot saved: {save_path}")

    def plot_pairplot(self, filename="pairplot.png"):
        numeric_data = self.data.select_dtypes(include=["int64", "float64"])
        sample_size = min(2000, len(numeric_data))
        sample_data = numeric_data.sample(n=sample_size, random_state=42)
        pair_plot = sns.pairplot(sample_data)
        pair_plot.fig.suptitle("Pairplot of California Housing Dataset", y=1.02)
        save_path = self.plot_dir / filename
        pair_plot.savefig(save_path, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Pairplot saved: {save_path}")

    def plot_violin(self, column, filename=None):
        if filename is None:
            filename = f"{column}_violin.png"
        plt.figure(figsize=(8, 6))
        sns.violinplot(y=self.data[column])
        plt.ylabel(column)
        plt.title(f"Violin Plot of {column}")
        plt.tight_layout()
        save_path = self.plot_dir / filename
        plt.savefig(save_path)
        plt.show()
        plt.close()
        print(f"Violin plot saved: {save_path}")

    def plot_categorical_count(self, column, filename=None):
        if filename is None:
            filename = f"{column}_countplot.png"
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.data, x=column)
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.title(f"Count of {column}")
        plt.xticks(rotation=30)
        plt.tight_layout()
        save_path = self.plot_dir / filename
        plt.savefig(save_path)
        plt.show()
        plt.close()
        print(f"Count plot saved: {save_path}")

    def plot_target_distribution(self, target_column="median_house_value", filename="target_distribution.png"):
        plt.figure(figsize=(10, 6))
        plt.hist(self.data[target_column].dropna(), bins=50, edgecolor="black")
        plt.xlabel(target_column)
        plt.ylabel("Frequency")
        plt.title("Distribution of Median House Value")
        plt.tight_layout()
        save_path = self.plot_dir / filename
        plt.savefig(save_path)
        plt.show()
        plt.close()
        print(f"Target distribution saved: {save_path}")

    def plot_income_vs_house_value(self, filename="income_vs_house_value.png"):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data["median_income"], self.data["median_house_value"], alpha=0.4)
        plt.xlabel("Median Income")
        plt.ylabel("Median House Value")
        plt.title("Median Income vs Median House Value")
        plt.tight_layout()
        save_path = self.plot_dir / filename
        plt.savefig(save_path)
        plt.show()
        plt.close()
        print(f"Income vs house value plot saved: {save_path}")

    def plot_color_bar(self, filename="color_bar.png"):
        plt.figure(figsize=(12, 8))
        plt.scatter(
            self.data["longitude"],
            self.data["latitude"],
            c=self.data["median_house_value"],
            cmap="viridis",
            alpha=0.5
        )
        plt.colorbar(label="Median House Value")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("California Housing Prices by Location")
        save_path = self.plot_dir / filename
        plt.savefig(save_path)
        plt.show()
        plt.close()
        print(f"Color bar plot saved: {save_path}")

    def compare_deep_learning_models(
        self,
        model1_predictions,
        model2_predictions,
        y_test,
        model1_name="Model 1",
        model2_name="Model 2"
    ):
        plt.figure(figsize=(12, 6))
        plt.scatter(y_test, model1_predictions, alpha=0.5, label=model1_name)
        plt.scatter(y_test, model2_predictions, alpha=0.5, label=model2_name)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title("Comparison of Deep Learning Models")
        plt.legend()
        plt.tight_layout()
        save_path = self.plot_dir / "deep_learning_models_comparison.png"
        plt.savefig(save_path)
        plt.show()
        plt.close()
        print(f"Deep learning models comparison plot saved: {save_path}")


def main():
    print("=" * 60)
    print("CALIFORNIA HOUSING - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    data = load_data(DATA_PATH)

    print("\nFIRST 5 ROWS")
    print("=" * 60)
    print(data.head())
    print("\nLAST 5 ROWS")
    print("=" * 60)
    print(data.tail())
    print("\nDATASET INFORMATION")
    print("=" * 60)
    data.info()

    print("\nDATASET SHAPE")
    print("=" * 60)
    print(data.shape)

    print("\nCOLUMN NAMES")
    print("=" * 60)
    print(data.columns.tolist())

    print("\nMISSING VALUES BEFORE CLEANING")
    print("=" * 60)
    print(data.isnull().sum())

    print("\nDUPLICATE ROWS")
    print("=" * 60)
    print(data.duplicated().sum())

    data = preprocess_data(data)
    print("\nMISSING VALUES AFTER CLEANING")
    print("=" * 60)
    print(data.isnull().sum())

    print("\nCATEGORICAL COLUMNS")
    print("=" * 60)
    categorical_columns = data.select_dtypes(include=["object", "category"]).columns
    for column in categorical_columns:
        print(f"\n{column}")
        print(data[column].unique())

    print("\nSTATISTICAL SUMMARY")
    print("=" * 60)
    print(data.describe())

    visualization = Visualization(data, PLOT_DIR)

    print("\n")
    print("=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    visualization.plot_histogram("median_income")
    visualization.plot_correlation_heatmap()
    visualization.plot_scatter("median_income", "households")
    visualization.plot_box("median_income")
    visualization.plot_pairplot()
    visualization.plot_violin("median_income")
    
    if "ocean_proximity" in data.columns:
        visualization.plot_categorical_count("ocean_proximity")

    visualization.plot_target_distribution("median_house_value")
    visualization.plot_income_vs_house_value()
    visualization.plot_color_bar()

    np.random.seed(42)
    y_test_dummy = data["median_house_value"].sample(n=500, random_state=42).values
    model1_predictions_dummy = y_test_dummy + np.random.normal(0, 50000, size=500)
    model2_predictions_dummy = y_test_dummy + np.random.normal(0, 70000, size=500)

    visualization.compare_deep_learning_models(
        model1_predictions_dummy,
        model2_predictions_dummy,
        y_test_dummy,
        "ANN Model",
        "RNN Model"
    )

    print("\n")
    print("=" * 60)
    print("ALL VISUALIZATIONS COMPLETED")
    print("=" * 60)
    print(f"\nAll plots are saved inside:\n{PLOT_DIR}")


if __name__ == "__main__":
    main()