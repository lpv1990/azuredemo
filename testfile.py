import pandas as pd
import matplotlib.pyplot as plt 
from numpy import savetxt 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes 
from sklearn.ensemble import RandomForestRegressor

db = load_diabetes()
X = db.data
y = db.target
X_train, X_test, y_train, y_test = train_test_split(X, y)


n_estimators = 100
max_depth = 6
max_features = 3
    # which will come under 
    # Create and train model.
rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
rf.fit(X_train, y_train)
    
    # Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)  
    
print(predictions)
