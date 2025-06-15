from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scaler
model = load_model('weather_multifeature_lstm_model.h5')
scaler = joblib.load('multifeature_scaler.save')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    temp = hum = press = ''

    if request.method == 'POST':
        try:
            temp = float(request.form.get('temp'))
            hum = float(request.form.get('hum'))
            press = float(request.form.get('press'))

            # Create a dummy 24-step sequence with the same input
            input_sequence = np.array([[temp, hum, press]] * 24)
            input_scaled = scaler.transform(input_sequence)
            input_scaled = input_scaled.reshape(1, 24, 3)

            # Predict next step (output is 3 values)
            pred_scaled = model.predict(input_scaled)
            pred = scaler.inverse_transform(pred_scaled)[0]

            prediction = {
                'temp': f"{pred[0]:.2f} °C",
                'hum': f"{pred[1]:.2f}",
                'press': f"{pred[2]:.2f} mb"
            }

        except Exception as e:
            prediction = {'error': str(e)}

    return render_template('index.html', prediction=prediction,
                           temp=temp, hum=hum, press=press)

if __name__ == '__main__':
    app.run(debug=True)
