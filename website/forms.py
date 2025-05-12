from django import forms

class PredictionForm(forms.Form):
    # 30 input fields as float
    features = [f'feature_{i}' for i in range(30)]

    for i, name in enumerate(features):
        locals()[name] = forms.FloatField(label=f"Feature {i + 1}")
