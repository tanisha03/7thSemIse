import pandas as pd
df = pd.read_csv('data.csv')
print(df.head)

# data processing
df = df.drop('Unnamed: 0', axis=1)

#check for NA values
df.isna().sum()
df = df.fillna(df.median())
# We use median instead of mean because median is not affected by outliers, whereas mean is.

#handle categorical value - hot encode

#dependent and independent variables
df['quality'].value_counts()

# Quality is on a scale of 1 to 10, but our data only has samples for 3,4,5,6,7,8.

# Lets convert this into a binary classification problem for simplicity sake.

# Thus, we'll covert this scale of quality into 2 categories, ['low', 'high'] defined as <=5 is low, >5 is high.

def get_quality(x):
    if x <= 5:
        return 'low'
    else:
        return 'high'

df['quality'] = df['quality'].apply(lambda x: get_quality(x))
df['quality'].value_counts()
df.to_csv('data_cleaned.csv', index=False)