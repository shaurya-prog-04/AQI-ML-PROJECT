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

#feature selection
from sklearn.feature_selection import SequentialFeatureSelector, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

X = df.drop(columns=['AQI_Label'])
y = df['AQI_Label']

X = X.fillna(X.median())

mi_scores = pd.Series(mutual_info_classif(X, y, random_state=42), index=X.columns)

sns.heatmap(X.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('plot5_correlation.png')
plt.show()

corr = X.corr()
to_drop = set()
for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > 0.9:
            col_i = corr.columns[i]
            col_j = corr.columns[j]
            to_drop.add(col_i if mi_scores[col_i] < mi_scores[col_j] else col_j)

# Also drop features with near zero MI score (useless features)
low_mi = mi_scores[mi_scores < 0.01].index.tolist()
to_drop.update(low_mi)

X = X.drop(columns=to_drop)
print("Dropped:", to_drop)

mi_scores = pd.Series(mutual_info_classif(X, y, random_state=42), index=X.columns).sort_values(ascending=False)
print(mi_scores)

mi_scores.plot(kind='bar', color='steelblue')
plt.title('Feature Importance — Mutual Information')
plt.tight_layout()
plt.savefig('plot6_mutual_info.png')
plt.show()

model = LogisticRegression(max_iter=1000, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

max_features = min(8, len(X.columns) - 1)
scores = []
feature_subsets = []

for n in range(1, max_features + 1):
    sfs = SequentialFeatureSelector(model, n_features_to_select=n, direction='forward')
    sfs.fit(X_scaled, y)
    selected = X.columns[sfs.get_support()].tolist()
    score = cross_val_score(model, X_scaled[:, sfs.get_support()], y, cv=3).mean()
    scores.append(score)
    feature_subsets.append(selected)
    print(f"n={n} | score={score:.4f} | features={selected}")

pd.Series(scores, index=range(1, max_features + 1)).plot(kind='line', marker='o', color='steelblue')
plt.title('SFS — Number of Features vs Accuracy Score')
plt.xlabel('Number of Features')
plt.ylabel('Cross Validation Score')
plt.xticks(range(1, max_features + 1))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot7_sfs_scores.png')
plt.show()

best_n = scores.index(max(scores)) + 1
top_features = feature_subsets[best_n - 1]
print(f"\nBest number of features : {best_n}")
print(f"Final selected features : {top_features}")

X = X[top_features]