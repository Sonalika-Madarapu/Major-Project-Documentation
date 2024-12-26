from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

@app.route('/')
def home():
    return render_template("form.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        carat = float(request.form['carat'])
        cut = int(request.form['cut'])
        color = int(request.form['color'])
        clarity = int(request.form['clarity'])
        depth = float(request.form['depth'])
        table = float(request.form['table'])
        x = float(request.form['x'])
        y = float(request.form['y'])
        z = float(request.form['z'])

        # Prepare the input for the model
        features = np.array([[carat, cut, color, clarity, depth, table, x, y, z]])
        
        # Scale the features
        scaled_features = scaler.transform(features)

        # Predict the price
        predicted_price = model.predict(scaled_features)[0]
        
        return render_template("result.html", prediction=predicted_price)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
