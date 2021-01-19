import pandas as pd
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("CustomerData.csv")
df.head()

# Data Processing
print(df.isnull().sum())
df.gender.fillna(df.gender.mode()[0], inplace=True)
df.category.fillna(df.category.mode()[0], inplace=True)
df.age.fillna(int(df.age.mean()), inplace=True)
df["annual income (lakhs)"].fillna(df["annual income (lakhs)"].mean(), inplace=True)
print(df.isnull().sum())

# Data transformation

encoder = preprocessing.LabelEncoder()
df[["category", "purchase type ", "gender"]] = df[
    ["category", "purchase type ", "gender"]
].apply(encoder.fit_transform)
print(df.head())

scaler = preprocessing.MinMaxScaler()
df[
    ["spending score", "items purchased (monthly)", "annual income (lakhs)"]
] = scaler.fit_transform(
    df[["spending score", "items purchased (monthly)", "annual income (lakhs)"]]
)
print(df.head())
