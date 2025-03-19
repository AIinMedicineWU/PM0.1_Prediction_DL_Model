from itertools import chain
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import time
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class RegressionDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


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


def main():
    start_time = time.time()

    # Load and preprocess data
    df = pd.read_csv("upsampled-pm-3hours-final-all-linear.csv", encoding="utf8").dropna()
    df = df.replace("-", float("nan")).dropna()
    df['Month'] = LabelEncoder().fit_transform(df['Month'])
    df = df[['PM2_5G', 'Month', 'Temperature', 'Humidity', 'Wind_s', 'SD_Wind_s', 'Wind_d', 'SD_Wind_d', 'Rainfall',
             'PM0_1G']].dropna()

    method = 'upsampled-pm-final-all-spline-1'

    # Prepare features and target
    X = df[['PM2_5G', 'Month', 'Temperature', 'Humidity', 'Wind_s', 'SD_Wind_s', 'Wind_d', 'SD_Wind_d', 'Rainfall']]
    y = df[['PM0_1G']]
    X = pd.DataFrame(X).astype(float)
    y = pd.DataFrame(y).astype(float)

    X_dataset, y_dataset = np.array(X), np.array(y)

    # Hyperparameters
    k_folds = 10
    EPOCHS = 750
    BATCH_SIZE = 16
    NUM_FEATURES = len(X.columns)
    learning_rates = [0.0001]

    # Prepare device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Running on {device}')

    # K-Fold Cross Validation
    splits = KFold(n_splits=k_folds, shuffle=True, random_state=123)
    dataset = RegressionDataset(torch.from_numpy(X_dataset).float(), torch.from_numpy(y_dataset).float())

    # Results tracking
    loss_train_stats = []
    fold_result_stats = []
    best_model = None
    best_r2 = float('-inf')

    for i, learning_rate in enumerate(learning_rates):
        print(f"Training with Learning Rate: {learning_rate}")

        y_pred_list = []
        y_test = []

        for fold, (train_idx, valid_idx) in enumerate(splits.split(dataset)):
            print(f'Fold : {fold}')

            # Prepare data loaders
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            test_subsampler = torch.utils.data.SubsetRandomSampler(valid_idx)

            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=BATCH_SIZE, sampler=test_subsampler)

            # Initialize model
            model = MultipleRegression(NUM_FEATURES)
            model.apply(reset_weights)
            model.to(device)

            # Loss and Optimizer
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

            # Training
            for epoch in range(EPOCHS):
                train_epoch_loss = 0

                for x, y in trainloader:
                    x, y = x.to(device), y.to(device)
                    y_hat = model(x)
                    loss = criterion(y_hat, y)
                    train_epoch_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss_train_stats.append({
                    "FOLD": f"FOLD {fold}",
                    "EPOCHS": epoch,
                    "Loss": train_epoch_loss / len(trainloader)
                })

            # Validation
            with torch.no_grad():
                for i, data in enumerate(testloader, 0):
                    inputs, targets = data
                    inputs = inputs.to(device)
                    predicted = model(inputs)

                    y_pred_list.append(predicted.cpu().numpy().flatten())
                    y_test.append(targets.squeeze().numpy().flatten())

            # Metrics calculation
            y_pred_1Dlist = np.concatenate(y_pred_list)
            y_test_1Dlist = np.concatenate(y_test)

            mse = mean_squared_error(y_test_1Dlist, y_pred_1Dlist)
            rmse = root_mean_squared_error(y_test_1Dlist, y_pred_1Dlist)
            r_square = r2_score(y_test_1Dlist, y_pred_1Dlist)

            print(f"Fold {fold} Results:")
            print(f"Mean Squared Error: {mse}")
            print(f"Root Mean Squared Error: {rmse}")
            print(f"R^2 Score: {r_square}\n")

            # Track best model
            if r_square > best_r2:
                best_r2 = r_square
                best_model = model

            fold_result_stats.append({
                "FOLD": f"FOLD {fold}",
                "R2": r_square,
                "MSE": mse,
                "RMSE": rmse
            })

    # Overall Results
    df_fold_result = pd.DataFrame(fold_result_stats)
    print("Cross-Validation Results:")
    print(f"Average MSE: {np.mean(df_fold_result['MSE']):.4f}")
    print(f"Average RMSE: {np.mean(df_fold_result['RMSE']):.4f}")
    print(f"Average R^2: {np.mean(df_fold_result['R2']):.4f}")

    # Save best model
    os.makedirs('saved_models', exist_ok=True)
    model_save_path = f'saved_models/{method}_best_model.pth'
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'method': method,
        'mean_mse': np.mean(df_fold_result['MSE']),
        'mean_rmse': np.mean(df_fold_result['RMSE']),
        'mean_r2': np.mean(df_fold_result['R2'])
    }, model_save_path)
    print(f"Best model saved to {model_save_path}")

    # Execution time
    end_time = time.time()
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()