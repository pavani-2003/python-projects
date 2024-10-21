import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load your dataset
# Replace 'your_dataset.csv' with the path to your dataset file
data = pd.read_csv("C:\Users\pavani.k\OneDrive\Documents\mobile prices.csv.xlsx")

# Assuming your dataset has a 'target' column that contains the labels
X = data.drop('target', axis=1)  # Features
y = data['target']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

