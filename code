import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('city_day.csv')

print(df.shape)
print(df.head())
print(df.isnull().sum())
print(df['AQI_Bucket'].value_counts())

df.drop(columns=['City', 'AQI'], inplace=True)
df.dropna(subset=['AQI_Bucket'], inplace=True)

pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx',
                  'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']

for col in pollutant_cols:
    df[col].fillna(df[col].median(), inplace=True)

Q1 = df[pollutant_cols].quantile(0.25)
Q3 = df[pollutant_cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[pollutant_cols] < (Q1 - 1.5 * IQR)) |
          (df[pollutant_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

df['Date']      = pd.to_datetime(df['Date'])
df['Month']     = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Year']      = df['Date'].dt.year

def get_season(month):
    if month in [12, 1, 2]: return 0
    if month in [3,  4, 5]: return 1
    if month in [6,  7, 8]: return 2
    return 3

df['Season'] = df['Month'].apply(get_season)
df.drop(columns=['Date'], inplace=True)

le = LabelEncoder()
df['AQI_Label'] = le.fit_transform(df['AQI_Bucket'])
df.drop(columns=['AQI_Bucket'], inplace=True)

print(df.shape)
print(df.isnull().sum().any())
print(dict(enumerate(le.classes_)))

df['AQI_Label'].value_counts().plot(kind='bar', color='steelblue')
plt.title('AQI Category Distribution')
plt.xlabel('AQI Category')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plot1_distribution.png')
plt.show()

pd.read_csv('city_day.csv')[pollutant_cols].isnull().mean().mul(100).plot(kind='barh', color='tomato')
plt.title('Missing Values %')
plt.xlabel('Missing %')
plt.tight_layout()
plt.savefig('plot2_missing.png')
plt.show()

df['PM2.5'].plot(kind='hist', bins=40, color='steelblue', edgecolor='white')
plt.title('PM2.5 Distribution')
plt.xlabel('PM2.5')
plt.tight_layout()
plt.savefig('plot3_pm25.png')
plt.show()

df.groupby('Month')['AQI_Label'].mean().plot(kind='line', marker='o', color='purple')
plt.title('Average AQI Level by Month')
plt.xlabel('Month')
plt.ylabel('Mean AQI Label')
plt.tight_layout()
plt.savefig('plot4_monthly.png')
plt.show()

df.to_csv('city_day_clean.csv', index=False)
print("Saved: city_day_clean.csv")