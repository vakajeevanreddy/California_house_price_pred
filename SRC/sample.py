import pandas as pd

# 1. Load a sample row from your CSV
df = pd.read_csv(
    r"C:\Users\lenovo\OneDrive\Desktop\California_housing\Data\raw\california_housing.csv"
)
df = df.drop(columns=["median_house_value"])  # Drop target variable

# 2. Automatically generate Pydantic Fields from the first row
sample_row = df.iloc[0].to_dict()

print("class HousingInput(BaseModel):")
for col_name, val in sample_row.items():
    python_type = "str" if isinstance(val, str) else "float"
    print(f'    {col_name}: {python_type} = Field(..., example={repr(val)})')