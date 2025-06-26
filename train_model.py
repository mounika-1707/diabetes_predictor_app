import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Step 1: Load dataset
# For demo, using sample diabetes dataset from UCI (replace with actual CSV if needed)
data = pd.read_csv('diabetes.csv')

# Step 2: Split into features and labels
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 5: Save model to a file
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Step 6: Confirmation message
print("Model trained and saved as 'diabetes_model.pkl'")
