import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the data
data = pq.read_table('metro_interview.parquet').to_pandas()

# Perform EDA
print(data.head())

# Split the data
X = data[['bedrooms', 'bathrooms', 'area']]
y = data['revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
import joblib
joblib.dump(model, 'model.pkl')
