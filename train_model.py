import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
if 'Cabin' in df.columns:
    df = df.drop(columns=['Cabin'])

encoder = LabelEncoder()
df['Sex'] = encoder.fit_transform(df['Sex'])
df['Embarked'] = encoder.fit_transform(df['Embarked'])
df = df.fillna(df.median(numeric_only=True))

# For plotting only (keep real labels)
df_plot = df.copy()
df_plot['Sex'] = df_plot['Sex'].map({0: "Female", 1: "Male"})

plt.figure(figsize=(12,10))

plt.subplot(2,2,1)
df['Survived'].value_counts().plot(kind='bar')
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Count")

plt.subplot(2,2,2)
df_plot.groupby('Sex')['Survived'].mean().plot(kind='bar')
plt.title("Survival Rate by Gender")
plt.ylabel("Rate")

plt.subplot(2,2,3)
df[df['Survived']==1]['Age'].plot(kind='hist', alpha=0.6, label='Survived')
df[df['Survived']==0]['Age'].plot(kind='hist', alpha=0.6, label='Not Survived')
plt.legend()
plt.title("Age Distribution")
plt.xlabel("Age")

plt.subplot(2,2,4)
df.groupby('Pclass')['Survived'].mean().plot(kind='bar')
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Rate")

plt.tight_layout()
plt.show()

X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

log_model = LogisticRegression(max_iter=500)
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("Accuracy Score:", acc)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", cr)
