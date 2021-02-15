import pandas as pd
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

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

features = df[["age", "gender", "annual income (lakhs)"]]
target = df["purchase type "]

x_train, x_test, y_train, y_test = train_test_split(
    features, target, random_state=0, test_size=0.3
)

dataframe = df[["gender", "age", "category"]]

from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=5, affinity="euclidean")
agg.fit_predict(dataframe)
dataframe["cluster"] = agg.labels_

sns.scatterplot(data=dataframe, x=dataframe.cluster, y=dataframe.age)
plt.show()

sns.scatterplot(
    data=dataframe,
    x=dataframe.cluster,
    y=dataframe.gender.replace({0: "Male", 1: "Female"}),
)
plt.show()

sns.scatterplot(data=dataframe, x=dataframe.cluster, y=dataframe.category)
plt.show()


from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=3, min_samples=3)
clusters = dbscan.fit_predict(dataframe)
dataframe["dbscan cluster"] = clusters

sns.scatterplot(data=dataframe, x=dataframe["dbscan cluster"], y=dataframe["age"])
# sns.scatterplot(data=dataframe, x=dataframe["dbscan cluster"], y=dataframe["income"])
sns.scatterplot(
    data=dataframe,
    x=dataframe["dbscan cluster"],
    y=dataframe["gender"].replace({0: "F", 1: "M"}),
)

