from flask import Flask, request, jsonify, render_template_string
import numpy as np
import joblib
from xgboost import XGBRegressor

# # Load the saved model
# model = joblib.load("linear_regression_model.joblib")

# importing model
model = XGBRegressor()
model.load_model('model.json')

# Create Flask app
app = Flask(__name__, template_folder='templates')

# HTML template for user input
html_template = """
<!DOCTYPE html>
<html>
    <head>
        <title>Linear Regression Predictor</title>
    </head>
    <body>
        <h2>Enter Feature Values for Prediction</h2>
        <form action="/predict" method="post">
            <label for="feature1">Feature 1:</label><br>
            <input type="number" step="any" id="feature1" name="feature1" required><br>
            <label for="feature2">Feature 2:</label><br>
            <input type="number" step="any" id="feature2" name="feature2" required><br>
            <label for="feature3">Feature 3:</label><br>
            <input type="number" step="any" id="feature3" name="feature3" required><br>
            <label for="feature4">Feature 4:</label><br>
            <input type="number" step="any" id="feature4" name="feature4" required><br>
            <label for="feature5">Feature 5:</label><br>
            <input type="number" step="any" id="feature5" name="feature5" required><br>
            <label for="feature6">Feature 6:</label><br>
            <input type="number" step="any" id="feature6" name="feature6" required><br>
            <input type="submit" value="Predict">
        </form>
    </body>
</html>
"""

# Route to display the form
@app.route("/")
def home():
    return render_template_string(html_template)

# Route to handle prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Extract feature values from the form data
    features = [
        float(request.form[f"feature{i}"]) for i in range(1, 7)
    ]
    # Convert to a numpy array
    features = np.array(features).reshape(1, -1)
    # Make a prediction
    prediction = model.predict(features)[0]
    # Return the prediction as JSON
    return jsonify({"prediction": prediction})

if __name__ == "__main__":

    app.run(debug=True)
