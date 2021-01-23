import pandas as pd

df = pd.read_csv('input/car_data.csv')

print(df.head())

print(df.shape)
print(df['Seller_Type'].unique())

