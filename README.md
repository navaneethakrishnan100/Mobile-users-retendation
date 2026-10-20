import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files

uploaded = files.upload()
file_name = next(iter(uploaded))

df = pd.read_csv(file_name)

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

display(df.head())
 print("Missing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates()

for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include="object").columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nData Cleaning Completed!")
print("Dataset Shape:", df.shape)
print("Statistical Summary:")
display(df.describe())

print("\nAverage App Usage Time:")
print(df["App Usage Time (min/day)"].mean())

print("\nAverage Screen Time:")
print(df["Screen On Time (hours/day)"].mean())

print("\nUser Behavior Class Count:")
print(df["User Behavior Class"].value_counts().sort_index())
 plt.figure(figsize=(8, 5))

plt.hist(df["App Usage Time (min/day)"], bins=20)

plt.title("App Usage Time Analysis")
plt.xlabel("App Usage Time (minutes/day)")
plt.ylabel("Number of Users")

plt.show()
plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="User Behavior Class")

plt.title("User Behavior Class Distribution")
plt.xlabel("User Behavior Class")
plt.ylabel("Number of Users")

plt.show()
 plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Screen On Time (hours/day)",
    y="App Usage Time (min/day)",
    hue="User Behavior Class"
)

plt.title("Screen Time vs App Usage")
plt.show()
 plt.figure(figsize=(10, 7))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.drop(columns=["User ID"], errors="ignore").corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")
plt.show()
 from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# Remove User ID
X = df.drop(columns=["User ID", "User Behavior Class"])
y = df["User Behavior Class"]

# Convert categorical data to numbers
X = pd.get_dummies(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.show()
 # Save cleaned dataset
df.to_csv("cleaned_user_behavior_dataset.csv", index=False)
