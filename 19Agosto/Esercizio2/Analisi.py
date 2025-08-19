import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
# leggo il dataset
df = pd.read_csv('Esercizio2\AirQualityUCI.csv', sep=';')
df = df.dropna(subset=['PT08.S1(CO)'])

# Creo le colonne da usare come features
df["Time"] = pd.to_datetime(df["Time"], format="%H.%M.%S")
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

df['Hour'] = df['Time'].dt.hour

df['target'] = np.where(df['PT08.S1(CO)'] > df['PT08.S1(CO)'].mean(),
                        'Buona', 'Scarsa')

X = df[['Year','Month','Day','Hour']]
y = df['target']

# Divisione in Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)

# Creazione del modello Random Forest
rf = RandomForestClassifier(n_estimators=15, random_state=42, class_weight='balanced')

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Random Forest: \n",classification_report(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_pred, y_test))

# Creazione del modello Decision Tree
dt = DecisionTreeClassifier(max_depth=7, class_weight='balanced')
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("Decision Tree: \n", classification_report(y_test, y_pred_dt))
print("Confusion Matrix: \n", confusion_matrix(y_pred_dt, y_test))
