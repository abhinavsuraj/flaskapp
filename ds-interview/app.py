from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

app = Flask(__name__)

# Assuming you have trained and saved your model as 'model.pkl'
try:
    model = joblib.load('model.pkl')
except FileNotFoundError:
    # Dummy model if model.pkl doesn't exist
    data = pd.DataFrame({
        'bedrooms': [1, 2, 3],
        'bathrooms': [1, 2, 3],
        'area': [1000, 1500, 2000],
        'revenue': [100, 150, 200]
    })
    X = data[['bedrooms', 'bathrooms', 'area']]
    y = data['revenue']
    model = LinearRegression().fit(X, y)
    joblib.dump(model, 'model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    bedrooms = float(data.get('bedrooms'))
    bathrooms = float(data.get('bathrooms'))
    area = float(data.get('area'))

    prediction = model.predict([[bedrooms, bathrooms, area]])[0]

    result = {"predicted_revenue": prediction}
    return jsonify(result)

@app.route('/test', methods=['GET'])
def test():
    return "Flask is running"

if __name__ == '__main__':
    app.run(debug=True)
