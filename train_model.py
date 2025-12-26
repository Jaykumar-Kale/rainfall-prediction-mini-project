import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("data/rainfall.csv")
print("Initial shape:", df.shape)

# Clean data
df.dropna(inplace=True)
print("After cleaning:", df.shape)

# Separate features and target
X = df.drop("ANNUAL", axis=1)
y = df["ANNUAL"]

# One-hot encode categorical column
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

# Save model and feature columns
joblib.dump(model, "model/rainfall_model.pkl")
joblib.dump(X.columns, "model/feature_columns.pkl")

print("Model and feature columns saved successfully")
