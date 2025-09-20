# ML-Profit-Predictor

This is a simple web application that predicts company profit based on inputs like R&D Spend, Administration Cost, and Marketing Spend using a Random Forest Regressor trained on the 50_Startups.csv dataset.

The app is built using:

Python (Flask)

Scikit-learn

HTML + Bootstrap

JavaScript (Fetch API)

nstallation & Setup

Clone the repository

git clone https://github.com/yourusername/ProfitPredictor.git
cd ProfitPredictor


Create a virtual environment (optional but recommended)

python -m venv venv
venv\Scripts\activate  # On Windows
# OR
source venv/bin/activate  # On macOS/Linux


Install dependencies

pip install -r requirements.txt


Train the model (optional)

If not already done, run the training script:

python train_model.py


This will generate:

best_model.pkl

scaler.pkl

Make sure these files exist in the root directory of the project.

▶️ Run the Application
python app.py


Then open your browser and go to:

http://127.0.0.1:5000

📝 Usage

Enter values for:

R&D Spend

Administration Cost

Marketing Spend

Click the “Predict Profit” button.

The app will return the predicted profit using the trained model.

📦 Requirements

Add this to requirements.txt:

Flask==2.3.3
scikit-learn==1.7.2
numpy
pandas
joblib


✅ Adjust scikit-learn version to match your environment.

🛠 Tech Stack

Backend: Python, Flask

ML Model: Scikit-learn, Random Forest Regressor

Frontend: HTML5, Bootstrap 5, JavaScript (Fetch API)

📌 Notes

Warnings about InconsistentVersionWarning from scikit-learn are safe to ignore unless the model truly breaks.

Always keep your best_model.pkl and scaler.pkl in sync with your scikit-learn version to avoid compatibility issues.

🧑‍💻 Author

Samreen – [GitHub Profile](https://github.com/fathimasamreen)

🪪 License

This project is licensed under the MIT License.
