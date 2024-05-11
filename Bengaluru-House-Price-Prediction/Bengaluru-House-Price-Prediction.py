# Imports
import pandas as pd
from math import floor
from matplotlib import rcParams as rcP
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import utils_Bengaluru

df = pd.read_csv('Bengaluru_House_Data.csv')
df = df.drop('society', axis='columns')

# Data Cleaning Process
balcony_median = float(floor(df.balcony.median()))
bath_median = float(floor(df.bath.median()))
df.balcony = df.balcony.fillna(balcony_median)
df.bath = df.bath.fillna(bath_median)
df = df.dropna()
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))
df = df.drop('size', axis='columns')
df['new_total_sqft'] = df.total_sqft.apply(utils_Bengaluru.convert_sqft_to_num)
df = df.drop('total_sqft', axis='columns')
df = df.dropna()

# Feature Engineering
df1 = df.copy()
df1['price_per_sqft'] = (df1['price']*100000)/df1['new_total_sqft']
locations = list(df['location'].unique())
df1.location = df1.location.apply(lambda x: x.strip())
location_stats = df1.groupby('location')['location'].agg('count').sort_values(ascending=False)
locations_less_than_10 = location_stats[location_stats<=10]
df1.location = df1.location.apply(lambda x: 'other' if x in locations_less_than_10 else x)
dates = df1.groupby('availability')['availability'].agg('count').sort_values(ascending=False)
dates_not_ready = dates[dates<10000]
df1.availability = df1.availability.apply(lambda x: 'Not Ready' if x in dates_not_ready else x)

# Removing Outliers
df2 = df1[~(df1.new_total_sqft/df1.bhk<300)]
df3 = utils_Bengaluru.remove_pps_outliers(df2)
df4 = utils_Bengaluru.remove_bhk_outliers(df3)
df5 = df4[df4.bath<(df4.bhk+2)]

# Model Building
df6 = df5.copy()
df6 = df6.drop('price_per_sqft', axis='columns')
dummy_cols = pd.get_dummies(df6.location).drop('other', axis='columns')
df6 = pd.concat([df6,dummy_cols], axis='columns')
dummy_cols = pd.get_dummies(df6.availability).drop('Not Ready', axis='columns')
df6 = pd.concat([df6,dummy_cols], axis='columns')
dummy_cols = pd.get_dummies(df6.area_type).drop('Super built-up  Area', axis='columns')
df6 = pd.concat([df6,dummy_cols], axis='columns')
df6.drop(['area_type','availability','location'], axis='columns', inplace=True)
X = df6.drop('price', axis='columns')
y = df6['price']

# Model Training
#print(utils_Bengaluru.find_best_model(X, y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20)
model = LinearRegression(n_jobs=True)
model.fit(X_train.values, y_train)
score = model.score(X_test, y_test)
print("The result of Linear Regression is {:.2f}%".format(score*100))

# Inference
print(utils_Bengaluru.prediction(model, X, '1st Block Jayanagar', 2, 2, 2, 1000, 'Built-up  Area', 'Ready To Move'))
print(utils_Bengaluru.prediction(model, X, '1st Phase JP Nagar', 2, 2, 2, 1000, 'Super built-up  Area', 'Ready To Move'))
print(utils_Bengaluru.prediction(model, X, '1st Phase JP Nagar', 2, 3, 2, 2000, 'Plot  Area', 'Not Ready'))