import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import xgboost as xgb

# 1. Load Data
data = pd.read_csv(
    r"C:\Users\lenovo\OneDrive\Desktop\California_housing\Data\raw\california_housing.csv"
)
#2. ADv Features
data["rooms_per_house_hold"] = data["total_rooms"] / data["households"]
data["bedrooms_per_room"] = data["total_bedrooms"]/data["total_rooms"]
data["population_per_household"] = data["population"]/data["households"]
#3. Distance to major econmic centers
data["dist_to_LA"] = np.sqrt((data["latitude"] - 34.05) ** 2 + (data["longitude"] - (-118.24))**2)
data["dist_to_SF"] = np.sqrt((data["latitude"]-37.77)**2+(data["longitude"]-(-122.41))**2)
#4.Spatial Clustering
kmeans = KMeans(n_clusters=12,random_state=42,n_init=10)
data["geo_cluster"] = kmeans.fit_predict(data[["latitude","longitude"]])

data.replace([np.inf,-np.inf],np.nan,inplace=True)

X = data.drop("median_house_value",axis=1)
Y = np.log1p(data['median_house_value']) 

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

categoric_cols = X.select_dtypes(include=['object','string']).columns
numeric_cols = X.select_dtypes(include=['int64','float64']).columns

preprocessor = ColumnTransformer(transformers=[('num',StandardScaler(),numeric_cols),('cat',OneHotEncoder(),categoric_cols)])

x_train_prep = preprocessor.fit_transform(x_train)
x_test_prep = preprocessor.transform(x_test)

xgb_reg = xgb.XGBRegressor(n_estimators = 1500,learning_rate = 0.003,max_depth = 8 ,subsample = 0.8,colsample_bytree=0.8,
    random_state=42,n_jobs=-1)
xgb_reg.fit(x_train_prep,y_train,eval_set=[(x_test_prep, y_test)],verbose=False)

y_pred_log = xgb_reg.predict(x_test_prep)

y_test_dollars = np.expm1(y_test)
y_pred_dollars = np.expm1(y_pred_log)

print("=== Performance Metrics ===")
print("R² (Log Space):", r2_score(y_test, y_pred_log))
print("R² (Dollar Space):", r2_score(y_test_dollars, y_pred_dollars))
print("RMSE (Dollar Space):", np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars)))
print("MAE (Dollar Space):", mean_absolute_error(y_test_dollars, y_pred_dollars))


