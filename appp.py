from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler once when the app starts
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
print("Model and scaler loaded from best_model.pkl and scaler.pkl")

def parse_float(value):
    # Remove commas, strip spaces and convert to float
    try:
        return float(value.replace(',', '').strip())
    except:
        raise ValueError(f"Invalid input for a numeric field: '{value}'")

@app.route('/')
def index():
    """Main page with prediction form"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        rd_spend = parse_float(request.form['rd_spend'])
        admin_cost = parse_float(request.form['admin_cost'])
        marketing_spend = parse_float(request.form['marketing_spend'])

        input_df = pd.DataFrame([{
            'R&D Spend': rd_spend,
            'Administration': admin_cost,
            'Marketing Spend': marketing_spend
        }])

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        # Return JSON response (your frontend JS expects this)
        return jsonify({
            'prediction': round(float(prediction), 2),
            'inputs': {
                'rd_spend': rd_spend,
                'admin_cost': admin_cost,
                'marketing_spend': marketing_spend
            }
        })

    except Exception as e:
        # Return JSON error
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
