import pandas as pd
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("CustomerData.csv")
df.head()

sns.distplot(df["age"])
plt.show()
sns.boxplot(x=df["age"])
plt.show()

# Removing outliers in age as person with age < 17 does not have a stable earning
df.drop(df[df["age"] < 17].index, inplace=True)
sns.distplot(df["age"])
plt.show()


sns.scatterplot(data=df, x=df.age, y=df["annual income (lakhs)"])
plt.show()
