from flask import Flask, render_template, request
import joblib
import numpy as np
from eda_visualize import generate_visualizations

app = Flask(__name__)

# Load trained model
model = joblib.load('models/sleep_disorder_model.pkl')


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    steps = None

    if request.method == 'POST':
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        sleep_duration = float(request.form['sleep_duration'])
        physical_activity = float(request.form['physical_activity'])
        stress_level = float(request.form['stress_level'])
        bmi_category = int(request.form['bmi_category'])
        systolic_bp = int(request.form['systolic_bp'])
        diastolic_bp = int(request.form['diastolic_bp'])
        heart_rate = int(request.form['heart_rate'])
        daily_steps = int(request.form['daily_steps'])
        smoking = int(request.form['smoking'])
        alcohol = int(request.form['alcohol'])

        input_features = np.array([[
            age, gender, sleep_duration, physical_activity,
            stress_level, bmi_category, systolic_bp,
            diastolic_bp, heart_rate, daily_steps,
            smoking, alcohol
        ]])

        pred = model.predict(input_features)[0]

        if pred == 0:
            prediction = "No Sleep Disorder"
            steps = "Maintain healthy sleep habits and lifestyle."
        elif pred == 1:
            prediction = "Mild Sleep Disorder"
            steps = "Improve sleep hygiene and manage stress."
        else:
            prediction = "Severe Sleep Disorder"
            steps = "Consult a healthcare professional."

    # Generate graphs every time page loads
    generate_visualizations()

    return render_template(
        'index.html',
        prediction=prediction,
        steps=steps
    )


if __name__ == "__main__":
    app.run()
