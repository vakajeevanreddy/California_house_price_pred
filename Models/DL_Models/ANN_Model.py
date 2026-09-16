import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,RobustScaler

data = pd.read_csv(
    r"C:\Users\lenovo\OneDrive\Desktop\California_housing\Data\raw\california_housing.csv"
)

#feature Engineering
data["rooms_per_house_hold"] = data["total_rooms"] / data["households"]
data["bedrooms_per_room"] = data["total_bedrooms"]/data["total_rooms"]
data["Population_per_house_hold"] = data["population"]/data["households"]

#Define X and Y
X = data.drop("median_house_value", axis=1)
y_raw = data["median_house_value"]
y = np.log1p(y_raw)

y_log_min,y_log_max = y.min(),y.max()

#split the Data
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
#pre processing pipelines
categoric_cols = X.select_dtypes(include=['object','string']).columns
numeric_cols = X.select_dtypes(include=[np.number]).columns
#build pipeline
num_pipeline = Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",RobustScaler())])
#category pipeline
cat_pipeline = Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                         ("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False))
                         ])
preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline , numeric_cols),("cat",cat_pipeline,categoric_cols)
])
x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)
# build the model
ann_model = Sequential(
    [
        Dense(128,activation="relu",kernel_regularizer=l2(1e-4),input_shape=(x_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64,activation="relu",kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32,activation="relu"),
        Dense(1,activation="linear"),

    ]
)
ann_model.summary()
#compile the model
ann_compile = ann_model.compile(optimizer=Adam(learning_rate=0.005),loss="huber",metrics=["mae"])
#implement Early stoppings
early_stoppings = EarlyStopping(monitor="val_loss",patience=20,restore_best_weights=True)
#reduce the learning rate
reduce_lr = ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=5,min_lr=1e-5)
#run the model
ann_history = ann_model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    callbacks=[early_stoppings, reduce_lr],
    verbose=1,
)
#evaluate with Target clipping
y_log_pred = ann_model.predict(x_test).flatten()
y_log_pred_clipped = np.clip(y_log_pred, y_log_min, y_log_max)

y_pred_dollar = np.expm1(y_log_pred_clipped)
y_test_dollar = np.expm1(y_test)

print("\n=== Performance Metrics ===")
print("R² (Log Space):    ", r2_score(y_test, y_log_pred_clipped))
print("R² (Dollar Space): ", r2_score(y_test_dollar, y_pred_dollar))
print("RMSE (Dollar Space):", np.sqrt(mean_squared_error(y_test_dollar, y_pred_dollar)))
print("MAE (Dollar Space): ", mean_absolute_error(y_test_dollar, y_pred_dollar))
