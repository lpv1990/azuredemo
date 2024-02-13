import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt 
from numpy import savetxt 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

db = load_diabetes()
X = db.data
y = db.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
print("hello")
expermentname="/Users/l.prasanna.velaga@accenture.com/demoflow"
project_id=1401579914566548
mlflow.set_experiment(experiment_id=project_id)

with mlflow.start_run():
    n_estimators = 100
    max_depth = 6
    max_features = 3
    # which will come under 
    # Create and train model.
    rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
    rf.fit(X_train, y_train)
    
    # Use the model to make predictions on the test dataset.
    predictions = rf.predict(X_test)  
    
    #print(predictions)
    
    mlflow.sklearn.log_model(rf, "RandomForestRegressor")
    

    mse=mean_squared_error(y_test,predictions)  # what are things you will do to predic 
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2_score = r2_score(y_test, predictions)

    mlflow.autolog(log_models=True) 
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2_score)
    
