import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Titanic dataset
def load_dataset():
    data = pd.read_csv('Titanic-Dataset.csv')
    return data

# Perform data preprocessing
def preprocess_data(data):
    # Drop irrelevant columns
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    # Handle missing values
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Fare'].fillna(data['Fare'].mean(), inplace=True)

    # Convert categorical features to numeric using label encoding
    le = LabelEncoder()
    data['Sex'] = le.fit_transform(data['Sex'])

    return data

# Perform feature scaling
def scale_features(data):
    scaler = StandardScaler()
    data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])
    return data

# Train and evaluate the model
def train_and_evaluate_model(data):
    # Split the data into training and testing sets
    X = data.drop('Survived', axis=1)
    y = data['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a random forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_classifier.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return accuracy, report

# Main function
def main():
    # Load the dataset
    data = load_dataset()

    # Perform data preprocessing
    data = preprocess_data(data)

    # Perform feature scaling
    data = scale_features(data)

    # Train and evaluate the model
    accuracy, report = train_and_evaluate_model(data)

    # Print the results
    print("Classification Report:")
    print(report)
    print("Accuracy:", accuracy)

# Execute the main function
if __name__ == "__main__":
    main()
