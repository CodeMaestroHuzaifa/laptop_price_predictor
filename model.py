import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error,r2_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# df = pd.read_csv('Data set\laptops_train.csv')
df = pd.read_csv('data_for_model.csv')
#------------------------------------------------------------------------------#

X = df.drop(columns=['Price'])
Y = np.log(df['Price'])

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15,random_state=2)



step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              min_samples_split = 2,
                              min_samples_leaf=1,
                              warm_start=False,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,Y_train)

y_pred = pipe.predict(X_test)

print("R2 Score",(r2_score(Y_test,y_pred)))
print("MSE:",mean_absolute_error(Y_test,y_pred))

#------------------------------------------------------------------------------#