import requests
import json

# Sample wine quality data from the dataset
test_data = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
}

# Test the API endpoint
url = "http://localhost:8000/predict"
response = requests.post(url, json=test_data)

print("Status Code:", response.status_code)
print("Response JSON:")
print(json.dumps(response.json(), indent=2))
