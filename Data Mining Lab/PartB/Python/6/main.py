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

from sklearn.ensemble import BaggingClassifier

bag = BaggingClassifier(random_state=0)
bag.fit(x_train, y_train)
predicted = bag.predict(x_test)
probability = bag.predict_proba(x_test)[:, 1]
bag.score(x_test, y_test)

from sklearn import metrics

print(metrics.classification_report(y_test, predicted))
print("\nConfusion matrix :\n", metrics.confusion_matrix(y_test, predicted))
print("\nArea under the curve: ", metrics.roc_auc_score(y_test, predicted))
fpr, tpr, thresholds = metrics.roc_curve(y_test, probability)
plt.plot(fpr, tpr, color="orange")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.show()
