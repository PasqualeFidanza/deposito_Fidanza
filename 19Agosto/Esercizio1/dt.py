import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# Lettura del dataset
df = pd.read_csv("AEP_hourly.csv", parse_dates=['Datetime'])

# Creazione della variabile target e suddivisione della data
df['month'] = df['Datetime'].dt.month
df['day'] = df['Datetime'].dt.day
df['hour'] = df['Datetime'].dt.hour
df['year'] = df['Datetime'].dt.year

df['target'] = (df['AEP_MW'] > df['AEP_MW'].mean()).astype(int)

# Creazione di training e test set
X = df[['year', 'month', 'day', 'hour']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Creazione del modello Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)



print(df.head())