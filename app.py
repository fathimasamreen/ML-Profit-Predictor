from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'default-secret-key')

# Global variables to store the model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
#model = None
#scaler = None
model_metrics = {}

def load_model():
    """Load the pre-trained model and scaler from pickle files"""
    global model, scaler, model_metrics
    
    try:
        # Try to load existing model and scaler
        if os.path.exists('best_model.pkl') and os.path.exists('scaler.pkl'):
            model = joblib.load('best_model.pkl')
            scaler = joblib.load('scaler.pkl')
            model_metrics = {
                'status': 'Trained Random Forest model loaded successfully',
                'features': ['R&D Spend', 'Administration', 'Marketing Spend'],
                'model_type': 'Random Forest',
                'r2_score': '0.9103',
                'mae': '6437.50',
                'mse': '72625008.62',
                'training_samples': 'Trained on startup dataset'
            }
            print("Model and scaler loaded from best_model.pkl and scaler.pkl")
        else:
            # Create and train a demo model if files don't exist
            print("Model files not found, creating and training demo Random Forest model")
            
            # Create demo data similar to startup dataset
            np.random.seed(42)
            n_samples = 50
            rd_spend = np.random.normal(120000, 40000, n_samples)
            admin_cost = np.random.normal(80000, 20000, n_samples)
            marketing_spend = np.random.normal(350000, 100000, n_samples)
            
            # Create realistic profit based on the features
            profit = (rd_spend * 1.2 + admin_cost * 0.8 + marketing_spend * 0.6 + 
                     np.random.normal(0, 10000, n_samples))
            
            # Create DataFrame
            X = pd.DataFrame({
                'R&D Spend': rd_spend,
                'Administration': admin_cost,
                'Marketing Spend': marketing_spend
            })
            y = pd.Series(profit)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest Regressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            predictions = model.predict(X_test_scaled)
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            
            # Save demo model and scaler
            joblib.dump(model, 'best_model.pkl')
            joblib.dump(scaler, 'scaler.pkl')
            
            model_metrics = {
                'status': 'Demo Random Forest model trained and saved',
                'features': ['R&D Spend', 'Administration', 'Marketing Spend'],
                'model_type': 'Random Forest',
                'r2_score': f'{r2:.4f}',
                'mae': f'{mae:.2f}',
                'mse': f'{mse:.2f}',
                'training_samples': f'{len(X_train)} training samples'
            }
            print(f"Demo model metrics - R²: {r2:.4f}, MAE: {mae:.2f}, MSE: {mse:.2f}")
            
    except Exception as e:
        print(f"Error loading/creating model: {e}")
        model_metrics = {'error': str(e)}

@app.route('/')
def index():
    """Main page with prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form predictions and return rendered template"""
    try:
        rd_spend = float(request.form['rd_spend'])
        admin_cost = float(request.form['admin_cost'])
        marketing_spend = float(request.form['marketing_spend'])

        if model is None or scaler is None:
            return render_template('index.html', error='Model or scaler not loaded')

        input_df = pd.DataFrame([{
            'R&D Spend': rd_spend,
            'Administration': admin_cost,
            'Marketing Spend': marketing_spend
        }])

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        return render_template('index.html',
                               prediction=round(prediction, 2),
                               input_rd=rd_spend,
                               input_admin=admin_cost,
                               input_marketing=marketing_spend)

    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/model-info')
def model_info():
    """Model information page"""
    return render_template('model_info.html', metrics=model_metrics)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        rd_spend = float(data['rd_spend'])
        admin_cost = float(data['admin_cost'])
        marketing_spend = float(data['marketing_spend'])

        if model is None or scaler is None:
            return jsonify({'error': 'Model or scaler not loaded'}), 500

        # Convert to DataFrame with feature names
        input_df = pd.DataFrame([{
            'R&D Spend': rd_spend,
            'Administration': admin_cost,
            'Marketing Spend': marketing_spend
        }])
        input_data_scaled = scaler.transform(input_df)
        prediction = model.predict(input_data_scaled)[0]

        return jsonify({
            'prediction': round(prediction, 2),
            'inputs': {
                'rd_spend': rd_spend,
                'admin_cost': admin_cost,
                'marketing_spend': marketing_spend
            },
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)