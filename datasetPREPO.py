import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
df = pd.read_csv("ciciot2023_dataset.csv")

# Select essential features for real-time traffic monitoring
selected_features = [
    "Protocol Type", "Rate", "Srate", "Drate", "TCP", "UDP", "Tot size", "IAT"
]

# Convert multiclass labels to binary labels
df['label'] = df['label'].apply(lambda x: 1 if x != "BenignTraffic" else 0)

# Split data into features and labels
X = df[selected_features]
y = df['label']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save the preprocessed dataset for model training
pd.DataFrame(X_train).to_csv("X_train.csv", index=False)
pd.DataFrame(y_train).to_csv("y_train.csv", index=False)
pd.DataFrame(X_test).to_csv("X_test.csv", index=False)
pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

print("Dataset preprocessing complete!")
