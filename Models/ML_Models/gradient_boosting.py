import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error

#load the dataset
data = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\California_housing\Data\raw\california_housing.csv")
# fill missing values
data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].mean())
# seperate features and target
X = data.drop('median_house_value', axis = 1)
Y = data['median_house_value']
# Identify numeric and categorical columns
categorical_cols = X.select_dtypes(include = ['object', 'string']).columns
numeric_cols = X.select_dtypes(include = ['int64', 'float64']).columns
# preprocessor : scale numeric , one-hot encode categorical
preprocessor = ColumnTransformer(
    transformers=[
        ('num' , StandardScaler() , numeric_cols),
        ('cat' , OneHotEncoder(handle_unknown = 'ignore') , categorical_cols)]
)
#gred pipeline with GradientBoostingRegressor
gb_pipeline = Pipeline(steps = [('preprocessor', preprocessor),
                                ('regressor', GradientBoostingRegressor(random_state = 42))])
x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size = 0.2 , random_state = 42)
#perform hyperparameter tuning with GridSearchCV
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__learning_rate': [0.01, 0.1, 0.2],
    'regressor__max_depth': [3, 5, 7]
}
#implementing GridsearchCv
grid_search_gb = GridSearchCV(gb_pipeline, param_grid , cv = 3 , scoring = 'r2' , n_jobs = -1)
#fit GridSearchCV on training set
grid_search_gb.fit(x_train, y_train)
#perform predictions on test set
gb_pred = grid_search_gb.predict(x_test)
print("Mean Absolute Error: " , mean_absolute_error(y_test,gb_pred))
print("Mean Squared Error: " , mean_squared_error(y_test,gb_pred))
print("R2 Score: " , r2_score(y_test,gb_pred))