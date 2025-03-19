# Deep-Learning-Based Prediction System for Ultrafine Particulate Matter (PM0.1) Concentration Using Meteorological Factors

Ultrafine particulate matter (PM0.1) poses significant health risks due to its ability to penetrate deeply into the human body. This study explores the relationship between meteorological factors and PM0.1 concentrations using a deep-learning regression model. The model aims to predict PM0.1 levels accurately, achieving high performance with an R² of 92.52% and an RMSE of 0.26 µg/m³. This approach facilitates widespread monitoring of PM0.1, supporting preventive health decision-making.
## Project Structure

- `train_pm01_model.py`: Contains the code for training the multiple regression model.
- `predict_pm01.py`: Contains the multiple regression model for predicting PM0.1G levels.
- `upsampled-pm-3hours-final-all-linear.csv`: Dataset used for training and evaluation.
- `saved_models/upsampled-pm-final-all-spline-1_best_model.pth`: Saved model weights.

## Requirements
- Python 3.8+
- PyTorch
- pandas
- numpy
- scikit-learn

## Installation
You can install the required Python packages using the following command:

```pip install PyTorch pandas numpy scikit-learn```

## Usage

To train the model, run the following command:

```python train_pm01_model.py```

To predict PM0.1 levels, run the following command:

```python predict_pm01.py```



## Dataset
The dataset is located in the project directory and is named `upsampled-pm-3hours-final-all-linear.csv`. It contains the following columns:
- `PM2_5G`: PM2.5 concentration in µg/m³.
- `Temperature`: Temperature in °C.
- `Relative Humidity`: Relative humidity in %.
- `Wind_s`: Wind speed in m/s.
- `SD_Wind_s`: Standard deviation of wind speed in m/s.
- `Wind_d`: Wind direction in degrees.
- `SD_Wind_d`: Standard deviation of wind direction in degrees.
- `Rainfall`: Amount of rainfall.
- `Month`: Month of the year.

## Model Architecture
The model architecture consists of a feedforward Deep Learning with the following layers:
- Input layer: 9  features
- Hidden layers: [256, 512, 1024, 256, 128]
- Activation function: ReLU

##License
This project is licensed under the MIT License. See the [LICENSE]() file for more details.
