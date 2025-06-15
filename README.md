# 🌦️ Weather Prediction Web App using LSTM

A real-time weather forecasting application built as part of my internship at **Novanectar Services Pvt. Ltd.**

This app predicts the **next hour's Temperature, Humidity, and Pressure** using an LSTM (Long Short-Term Memory) neural network trained on historical weather data.

---

## 🚀 Features

- 🔮 **Real-time Predictions:**
  - 🌡️ Temperature (°C)
  - 💧 Humidity (0–1)
  - 🌬️ Pressure (millibars)
- 🧠 Based on 24-hour historical patterns
- 💻 User-friendly web interface with Flask
- 🔄 Real-time inputs with immediate prediction

---

## 🧰 Tech Stack

| Component        | Technology           |
|------------------|----------------------|
| ML Model         | LSTM (Keras/TensorFlow) |
| Web Framework    | Flask (Python)       |
| Data Handling    | Pandas, NumPy        |
| Visualization    | Matplotlib (for testing) |
| Normalization    | MinMaxScaler (scikit-learn) |
| Model I/O        | Joblib               |
| Frontend         | HTML + CSS (basic UI) |

---

## 📁 Project Structure

```
weather-lstm-app/
├── app.py                    # Flask web server
├── model_train.py           # Script to train LSTM model
├── model_test.py            # Script to test and plot model results
├── weather_lstm_model.h5    # Trained LSTM model
├── scaler.save              # Saved MinMaxScaler
├── weatherHistory.csv       # Kaggle dataset used for training
├── templates/
│   └── index.html          # Web UI template
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

---

## 🧪 How to Run

### 1. Clone the Project
```bash
git clone https://github.com/your-username/weather-lstm-app.git
cd weather-lstm-app
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Contents of `requirements.txt`:**
```txt
flask
numpy
pandas
matplotlib
scikit-learn
joblib
tensorflow
```

### 3. Train the Model
```bash
python model_train.py
```

### 4. Run the Web App
```bash
python app.py
```

Then visit: **http://127.0.0.1:5000/** in your browser.

---

## 📊 Dataset Used

- **Source:** Kaggle Weather History Dataset
- **Features used:**
  - Temperature (°C)
  - Humidity
  - Pressure (millibars)

---

## ✅ Sample Prediction Output

**User Input:**
- Temperature: 22.5 °C
- Humidity: 0.78
- Pressure: 1012.4 mb

**Prediction:**
- Next Hour Temperature: 22.63 °C
- Next Hour Humidity: 0.76
- Next Hour Pressure: 1012.70 mb

---

## 🔧 Model Architecture

The LSTM model is designed to:
- Accept 24 hours of historical weather data as input
- Process sequential patterns in temperature, humidity, and pressure
- Output predictions for the next hour's weather conditions
- Use MinMaxScaler for data normalization

---

## 📈 Future Enhancements

- [ ] Add more weather parameters (wind speed, precipitation)
- [ ] Implement longer-term forecasting (6-12 hours)
- [ ] Deploy to cloud platforms (AWS, Heroku, Render)
- [ ] Add interactive charts and visualizations
- [ ] Implement API endpoints for external integration

---

## 🙌 Acknowledgement

This project was completed as part of my internship at **Novanectar Services Pvt. Ltd.**

Big thanks to the team for their support and guidance!

---

## 📬 Contact

**Gokul Krishnan**
- 📧 Email: [Email](mailto:gk5139272@gmail.com)
- 🔗 LinkedIn: [Profile](https://www.linkedin.com/in/gokul-krishnan-yn-1633a9258/)
- 🔗 GitHub: [GitHub Profile](https://github.com/Krishjunier)

---
## ⭐ Show Your Support

If you found this project helpful, please give it a ⭐ on GitHub!
