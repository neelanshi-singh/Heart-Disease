from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load pre-trained model and scaler
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                         'exang', 'oldpeak', 'slope', 'ca', 'thal']

        input_data = [float(request.form[feature]) for feature in feature_names]
        input_data_scaled = scaler.transform([input_data])

        prediction = model.predict(input_data_scaled)[0]
        prediction_proba = model.predict_proba(input_data_scaled)[0]

        print(f"User Input: {input_data}")
        print(f"Prediction: {prediction}, Probability: {prediction_proba}")

        result_text = "No Heart Disease Detected ❤️" if prediction == 0 else "Heart Disease Detected 💔"
        return render_template('index.html', prediction_text=f'Prediction: {result_text}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

@app.route('/test')
def test():
    test_input = np.array([[65, 1, 3, 160, 300, 1, 2, 120, 1, 2.5, 2, 2, 7]])
    test_input_scaled = scaler.transform(test_input)

    prediction = model.predict(test_input_scaled)[0]
    prediction_proba = model.predict_proba(test_input_scaled)[0]

    print(f"Test Input: {test_input}")
    print(f"Prediction: {prediction}, Probability: {prediction_proba}")

    result_text = "Heart Disease Detected 💔" if prediction == 1 else "No Heart Disease Detected ❤️"
    return render_template('index.html', prediction_text=f'Prediction: {result_text}')

if __name__ == "__main__":
    app.run(debug=True)
