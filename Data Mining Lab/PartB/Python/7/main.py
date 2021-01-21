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
df["annual income (lakhs)"].fillna(df["annual income (lakhs)"].mean(),
                                   inplace=True)
print(df.isnull().sum())

# Data transformation

encoder = preprocessing.LabelEncoder()
df[["category", "purchase type ",
    "gender"]] = df[["category", "purchase type ",
                     "gender"]].apply(encoder.fit_transform)
print(df.head())

scaler = preprocessing.MinMaxScaler()
df[["spending score", "items purchased (monthly)",
    "annual income (lakhs)"]] = scaler.fit_transform(df[[
        "spending score", "items purchased (monthly)", "annual income (lakhs)"
    ]])
print(df.head())

features = df[["age", "gender", "annual income (lakhs)"]]
target = df["purchase type "]

# This is not required in this, as we are doing clustering
x_train, x_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    random_state=0,
                                                    test_size=0.3)

from sklearn.cluster import KMeans


#Add income too, it is a good parameter to form clusters (Not necessarily)
dataframe = df[['gender', 'age', 'category']]


kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit_predict(dataframe)
print("Cluster centers\n", kmeans.cluster_centers_)
dataframe["cluster"] = kmeans.labels_

#A better reprsentation of clusters would be this.
sns.scatterplot(data=dataframe,x = dataframe.age,y = dataframe.category,hue='cluster')
plt.show()

sns.scatterplot(data=dataframe, x=dataframe.cluster, y=dataframe.age)
plt.show()

sns.scatterplot(
    data=dataframe,
    x=dataframe.cluster,
    y=dataframe.gender.replace({
        0: "Male",
        1: "Female"
    }),
)
plt.show()

