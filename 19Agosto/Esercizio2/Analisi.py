import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('Esercizio2\AirQualityUCI.csv', sep=';')
print(df.head())

df["Time"] = pd.to_datetime(df["Time"], format="%H.%M.%S")
df['Hour'] = df['Time'].dt.hour

df['target'] = np.where(df['PT08.S1(CO)'] > df['PT08.S1(CO)'].mean(),
                        'Buona', 'Scarsa')

print(df.head())

X = df[['Hour']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

rf = RandomForestClassifier(n_estimators=10, random_state=42)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(classification_report(y_test, y_pred))