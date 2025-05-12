from django.shortcuts import render
from .forms import CancerPredictionForm
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'lr_model_bc.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler_bc.pkl')

# Load model and scaler once
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

def index(request):
    result = None

    if request.method == 'POST':
        form = CancerPredictionForm(request.POST)
        if form.is_valid():
            input_data = [form.cleaned_data[f'feature_{i}'] for i in range(30)]
            scaled_input = scaler.transform([input_data])
            prediction = model.predict(scaled_input)[0]
            result = "Benign" if prediction == 1 else "Malignant"
    else:
        form = CancerPredictionForm()

    return render(request, 'predictor/index.html', {'form': form, 'result': result})
