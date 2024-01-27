import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
# Loading dataset
dataset=pd.read_csv('FastagFraudDetection.csv')
dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])
# Extracting useful features from timestamp
dataset['Hour'] = dataset['Timestamp'].dt.hour
dataset['DayOfWeek'] = dataset['Timestamp'].dt.dayofweek
dataset['Month'] = dataset['Timestamp'].dt.month
# Applying One-hot encoding 
dataset = pd.get_dummies(dataset, columns=['Vehicle_Type', 'Lane_Type'])
# Label encoding
le = LabelEncoder()
dataset['Fraud_indicator'] = le.fit_transform(dataset['Fraud_indicator'])
# Feature Extraction Using Haverin Distance 
reference_point = (13.059816123454882, 77.77068662374292)

dataset['distance_from_city_center'] = dataset['Geographical_Location'].apply(
    lambda x: geodesic(reference_point, tuple(map(float, x.split(',')))).kilometers
)
scaler = MinMaxScaler()
dataset[['Vehicle_Speed', 'Transaction_Amount', 'Amount_paid']] = scaler.fit_transform(
    dataset[['Vehicle_Speed', 'Transaction_Amount', 'Amount_paid']]
)
# Computing the correlation matrix 
numeric_columns = dataset.select_dtypes(include=np.number).columns
correlation_matrix = dataset[numeric_columns].corr()
# Setting a correlation threshold of 0.1
correlation_threshold = 0.1
# Selecting features with absolute correlation above the threshold
selected_features = correlation_matrix[abs(correlation_matrix['Fraud_indicator']) > correlation_threshold].index
# Keeping only the selected features in the dataset
print(selected_features)
dataset = dataset[selected_features]
# Handling NaN values by filling with mean value
dataset.fillna(dataset.mean(), inplace=True)
"""# Plot a heatmap to visualize correlations
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix")
plt.show()"""
# Handling Nan Values 
dataset.fillna(dataset.mean(), inplace=True)  
# Spliting Data into features and Labels
x = dataset.drop('Fraud_indicator', axis=1)
y = dataset['Fraud_indicator']
# Splitting training adn Testing Dataset 80% for training and 20% for testing 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Print class distribution in the original dataset
print("Class Distribution in Original Dataset:")
print(y_train.value_counts())
# Oversampling using RandomOverSampler
oversampler = RandomOverSampler(random_state=42)
x_train_resampled, y_train_resampled = oversampler.fit_resample(x_train, y_train)
# Printing class distribution after oversampling
print("\nClass Distribution after Oversampling:")
print(y_train_resampled.value_counts())
# Training the model on the resampled data
model = BalancedRandomForestClassifier(random_state=42)
model.fit(x_train_resampled, y_train_resampled)
# Making predictions on the test set
y_pred = model.predict(x_test)
# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Saving the model 
joblib.dump(model, 'model_FasttagFraudDetection.pkl')
