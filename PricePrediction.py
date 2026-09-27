import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error
import joblib

df = pd.read_csv('D:\\HousePrice\\Property1.csv')

features = ['city', 'location', 'property_type', 'bedrooms', 'baths', 'area_marla', 'purpose']
target = 'price'

df= df[df['purpose']=='For Sale']

df_model = df[features + [target]].dropna().copy()


def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df_clean = remove_outliers(df_model, 'price')

df_clean = remove_outliers(df_clean, 'area_marla')


#plt.figure(figsize=(6,4))
#sns.boxplot(x=df_clean['price'])
#plt.title('outliers')
#plt.show()

encoders = {}
cat_col= ['city', 'location', 'property_type']
for col in cat_col:
    le=LabelEncoder()
    df_clean[col]=le.fit_transform(df_clean[col].astype(str))
    encoders[col]=le

X=df_clean.drop(columns=['purpose', 'price'])
y=df_clean['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=60, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_predict= model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_predict))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_predict))

print("\nModel are trained successfully")

joblib.dump(model, 'model.pkl', compress=3)
joblib.dump(encoders, 'encoders.pkl')

print("\nModel and encoders are saved successfully")
