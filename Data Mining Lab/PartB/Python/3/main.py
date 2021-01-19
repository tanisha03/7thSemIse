import pandas as pd
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from apyori import apriori


df1 = pd.read_csv('association_analysis.csv')
df1.head()

df1.drop(['tid'], axis = 1, inplace = True)

# Converting dataframe to a list of lists containing items

records = []
for i in range(len(df1)):
    record = []
    for j in range(len(df1.columns)):
        if df1.values[i, j]:
            record.append(df1.columns[j])
    records.append(record)

print(records[:3])


min_sup = 0.03
min_confidence = 0.7

rules = apriori(records, min_support = min_sup, min_confidence = min_confidence)
rules = list(rules)


for rule in rules:
    items = [i for i in rule[0]]
    print("Rule : ",items, "Support :", rule[1], "Confidence : ", rule[2][0][2])