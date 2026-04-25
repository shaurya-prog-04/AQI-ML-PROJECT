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

#train-val-test-split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

print("Train size     :", X_train.shape)
print("Validation size:", X_val.shape)
print("Test size      :", X_test.shape)
print("\nTrain label distribution:\n", y_train.value_counts())
print("\nVal label distribution:\n",   y_val.value_counts())
print("\nTest label distribution:\n",  y_test.value_counts())

#model training

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

classes = ['Good', 'Moderate', 'Poor', 'Satisfactory', 'Severe', 'Very Poor']

lr  = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)

lr.fit(X_train, y_train)
knn.fit(X_train, y_train)
svm.fit(X_train, y_train)

lr_knn  = VotingClassifier(estimators=[('lr', lr), ('knn', knn)], voting='soft')
lr_svm  = VotingClassifier(estimators=[('lr', lr), ('svm', svm)], voting='soft')
knn_svm = VotingClassifier(estimators=[('knn', knn), ('svm', svm)], voting='soft')
all3    = VotingClassifier(estimators=[('lr', lr), ('knn', knn), ('svm', svm)], voting='soft')

lr_knn.fit(X_train, y_train)
lr_svm.fit(X_train, y_train)
knn_svm.fit(X_train, y_train)
all3.fit(X_train, y_train)

models = {
    'Logistic Regression' : lr,
    'KNN'                 : knn,
    'SVM'                 : svm,
    'LR + KNN'            : lr_knn,
    'LR + SVM'            : lr_svm,
    'KNN + SVM'           : knn_svm,
    'LR + KNN + SVM'      : all3
}

results = {}
for name, model in models.items():
    val_preds = model.predict(X_val)
    report    = classification_report(y_val, val_preds, target_names=classes, output_dict=True)
    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(f"Train Accuracy     : {accuracy_score(y_train, model.predict(X_train)):.4f}")
    print(f"Validation Accuracy: {accuracy_score(y_val, val_preds):.4f}")
    print(classification_report(y_val, val_preds, target_names=classes))
    results[name] = {
        'Accuracy' : round(accuracy_score(y_val, val_preds), 4),
        'Precision': round(report['weighted avg']['precision'], 4),
        'Recall'   : round(report['weighted avg']['recall'], 4),
        'F1 Score' : round(report['weighted avg']['f1-score'], 4)
    }

results_df = pd.DataFrame(results).T
print("\nFULL VALIDATION COMPARISON TABLE")
print(results_df)

best_accuracy = results_df['Accuracy'].max()
results_df['Diff from Best %'] = ((best_accuracy - results_df['Accuracy']) / best_accuracy * 100).round(2)

print("\nACCURACY DIFFERENCE FROM BEST MODEL")
print(results_df[['Accuracy', 'Diff from Best %']])

results_df['Accuracy'].plot(kind='bar', color='steelblue')
for i, (acc, diff) in enumerate(zip(results_df['Accuracy'], results_df['Diff from Best %'])):
    plt.text(i, acc + 0.005, f'-{diff}%', ha='center', fontsize=8, color='red')

plt.title('All Models — Accuracy Comparison with % Difference from Best')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('plot8_validation_comparison.png')
plt.show()

best = results_df['Accuracy'].idxmax()
print(f"\nBest model : {best}")
print(results_df.loc[best])

chosen_model_name = 'KNN'
chosen_model      = models[chosen_model_name]

print(f"\nChosen model for final evaluation: {chosen_model_name}")
print(f"Accuracy difference from best: {results_df.loc[best, 'Accuracy'] - results_df.loc[chosen_model_name, 'Accuracy']:.4f}")

#Testing

from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, accuracy_score, classification_report
from sklearn.preprocessing import label_binarize
val_preds   = chosen_model.predict(X_val)
test_preds  = chosen_model.predict(X_test)
test_probs  = chosen_model.predict_proba(X_test)

print(classification_report(y_test, test_preds, target_names=classes))

ConfusionMatrixDisplay.from_predictions(
    y_test, test_preds, display_labels=classes, colorbar=False)
plt.title(f'Confusion Matrix — {chosen_model_name}')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('plot9_confusion_matrix.png')
plt.show()

y_bin = label_binarize(y_test, classes=list(range(6)))
for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_bin[:, i], test_probs[:, i])
    plt.plot(fpr, tpr, label=f'{cls} (AUC={auc(fpr,tpr):.2f})')
plt.plot([0,1],[0,1],'k--')
plt.title(f'ROC Curve — {chosen_model_name}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig('plot10_roc_curve.png')
plt.show()

val_report  = classification_report(y_val,  val_preds,  target_names=classes, output_dict=True)
test_report = classification_report(y_test, test_preds, target_names=classes, output_dict=True)

comparison_df = pd.DataFrame({
    'Validation' : {
        'Accuracy' : round(accuracy_score(y_val,  val_preds),  4),
        'Precision': round(val_report['weighted avg']['precision'],  4),
        'Recall'   : round(val_report['weighted avg']['recall'],     4),
        'F1 Score' : round(val_report['weighted avg']['f1-score'],   4)
    },
    'Test' : {
        'Accuracy' : round(accuracy_score(y_test, test_preds), 4),
        'Precision': round(test_report['weighted avg']['precision'], 4),
        'Recall'   : round(test_report['weighted avg']['recall'],    4),
        'F1 Score' : round(test_report['weighted avg']['f1-score'],  4)
    }
})

print("\nValidation vs Test Comparison")
print(comparison_df)

comparison_df.plot(kind='bar', color=['steelblue', 'tomato'])
plt.title(f'Validation vs Test — {chosen_model_name}')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('plot11_val_vs_test.png')
plt.show()

print(f"\nTest Accuracy : {accuracy_score(y_test, test_preds):.4f}")
print(f"Test Accuracy : {accuracy_score(y_test, test_preds)*100:.2f}%")