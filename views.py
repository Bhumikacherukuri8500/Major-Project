import os
import joblib
import numpy as np
from django.shortcuts import render

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model components
model = joblib.load(os.path.join(BASE_DIR, 'model/iomt_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'model/scaler.pkl'))
label_encoder = joblib.load(os.path.join(BASE_DIR, 'model/label_encoder.pkl'))

def predict_view(request):
    prediction = None

    if request.method == 'POST':
        heart_rate = float(request.POST['heart_rate'])
        spo2 = float(request.POST['spo2'])
        systolic = float(request.POST['systolic'])
        diastolic = float(request.POST['diastolic'])
        temperature = float(request.POST['temperature'])

        input_data = np.array([[heart_rate, spo2, systolic, diastolic, temperature]])
        input_scaled = scaler.transform(input_data)

        pred = model.predict(input_scaled)
        prediction = label_encoder.inverse_transform(pred)[0]

    return render(request, 'predict.html', {'prediction': prediction})
