from flask import Flask, render_template, request
import numpy as np
import joblib
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the trained model
model = joblib.load('bigmart_model')  # Ensure this file exists in your project directory

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    item_mrp = float(request.form['item_mrp'])
    outlet_id = float(request.form['outlet_id'])
    outlet_size = float(request.form['outlet_size'])
    outlet_type = float(request.form['outlet_type'])
    outlet_age = float(request.form['outlet_age'])

    start_date = request.form['start_date']
    end_date = request.form['end_date']

    # Calculate the date range
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    delta_days = (end - start).days + 1

    # Generate predictions for each day in the date range
    predictions = []
    for i in range(delta_days):
        current_date = start + timedelta(days=i)
        dynamic_outlet_age = outlet_age + (i / 365)

        input_features = np.array([[item_mrp, outlet_id, outlet_size, outlet_type, dynamic_outlet_age]])
        base_pred = model.predict(input_features)[0]

        # Add realistic fluctuation to simulate daily variation (±5%)
        fluctuation_percentage = np.random.uniform(-0.05, 0.05)  # -5% to +5%
        final_pred = base_pred * (1 + fluctuation_percentage)

        margin = 714.42
        predictions.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'predicted_sales': float(round(final_pred, 2)),
            'range': f"{round(final_pred - margin, 2)} to {round(final_pred + margin, 2)}"
        })

    total_sales = float(sum(p['predicted_sales'] for p in predictions))

    return render_template('result.html', predictions=predictions, total=round(total_sales, 2))
