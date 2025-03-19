import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder


class MultipleRegression(nn.Module):
    def __init__(self, num_features):
        super(MultipleRegression, self).__init__()
        self.input = nn.Linear(num_features, 256)
        self.hidden_1 = nn.Linear(256, 512)
        self.hidden_2 = nn.Linear(512, 1024)
        self.hidden_3 = nn.Linear(1024, 256)
        self.hidden_4 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input(x))
        x = self.relu(self.hidden_1(x))
        x = self.relu(self.hidden_2(x))
        x = self.relu(self.hidden_3(x))
        x = self.relu(self.hidden_4(x))
        x = self.output(x)
        return x


model_path = "saved_models\\upsampled-pm-final-all-spline-1_best_model.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
num_features = 9
model = MultipleRegression(num_features)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)

model.eval()




def predict_pm01(input_data):

    features = ['PM2_5G', 'Month', 'Temperature', 'Humidity', 'Wind_s', 'SD_Wind_s', 'Wind_d', 'SD_Wind_d', 'Rainfall']
    X = input_data[features].astype(float)

    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(X_tensor)

    predictions_np = predictions.cpu().numpy()

    return predictions_np

sample_data = pd.DataFrame({
    'PM2_5G': [25.0],
    'Month': [5],
    'Temperature': [28.5],
    'Humidity': [65.0],
    'Wind_s': [2.5],
    'SD_Wind_s': [0.8],
    'Wind_d': [180.0],
    'SD_Wind_d': [15.0],
    'Rainfall': [0.0]
})

sample_prediction = predict_pm01(sample_data)
print(f"Predicted PM0.1G for sample data: {sample_prediction[0][0]:.4f}")