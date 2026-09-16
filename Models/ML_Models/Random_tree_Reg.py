import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
# pipeline with RandomForestrand
rf_pipeline = Pipeline(steps = [('preprocessor' , preprocessor),
                                ('rand' , RandomForestRegressor(random_state = 42,n_estimators = 500))])
# train and test split
x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size = 0.2 , random_state = 42)
# hyperparameter tuning with GridSearchCV
param_grid = {
    'rand__n_estimators': [500],
    'rand__max_depth': [None, 10],
    'rand__min_samples_split': [2],
    'rand__min_samples_leaf': [1]
}

#implementing GridsearchCv
grid_search_rf = GridSearchCV(rf_pipeline, param_grid , cv = 3 , scoring = 'r2' , n_jobs = -1)
#fit GridSearchCV on training set
grid_search_rf.fit(x_train, y_train)
#perform predictions on test set
rand_pred = grid_search_rf.predict(x_test)
print("Mean Absolute Error: " , mean_absolute_error(y_test,rand_pred))
print("Mean Squared Error: " , mean_squared_error(y_test,rand_pred))
print("R2 Score: " , r2_score(y_test,rand_pred))