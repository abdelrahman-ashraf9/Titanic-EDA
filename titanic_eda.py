import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(os.path.dirname(__file__))

df = pd.read_csv("titanic.csv")

# Data Exploration
print(df.head())

print("#" * 50)

print(df.shape)

print("#" * 50)

df.info()

print("#" * 50)

print(df.describe())

print("#" * 50)

plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=20)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")

plt.show()
print(df.groupby("Sex")["Survived"].mean())

print("#" * 50)

plt.figure(figsize=(8, 5))
survival_rate = df.groupby("Sex")["Survived"].mean()

survival_rate.plot(kind="bar")

plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")

plt.show()

print(df["Pclass"].value_counts())

print("#" * 50)

print(df.groupby("Pclass")["Survived"].mean())

print("#" * 50)

print(df.groupby("Pclass")["Age"].mean())

print("#" * 50)

print(df.groupby("Pclass")["Sex"].value_counts())

print("#" * 50)

print(df.groupby(["Pclass", "Sex"])["Survived"].mean())

# # Data Visualization 
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Pclass")

plt.title("Passengers by Class")
plt.show()

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="Sex")

plt.title("Passengers by Gender")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Pclass", hue="Survived")

plt.title("Survival by Passenger Class")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Sex", hue="Survived")

plt.title("Survival by Gender")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, y="Fare")

plt.title("Fare Distribution")
plt.show()

plt.figure(figsize=(10, 8))

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")

plt.title("Correlation Matrix")
plt.show()
# Data Cleaning
df["Age"] = df["Age"].fillna(df["Age"].median())

print(df["Age"].isnull().sum())

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df.drop(columns=["Cabin"], inplace=True)

df.info()
print("Data Cleaning Completed Successfully!")
df.to_csv("titanic_cleaned.csv", index=False)