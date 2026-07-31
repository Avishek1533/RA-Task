import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

class TimeSeriesSequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class PyTorchLSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(PyTorchLSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Take output from last time step
        return out

def create_sequences(scaled_data, seq_length=24):
    xs, ys = [], []
    for i in range(len(scaled_data) - seq_length):
        x = scaled_data[i:(i + seq_length)]
        y = scaled_data[i + seq_length, 0] # target demand
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_pytorch_lstm(train_df, test_df, model_save_path, scaler_save_path, seq_length=24, epochs=15, batch_size=128, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Target normalization
    scaler = MinMaxScaler()
    train_vals = train_df[['demand_mw']].values
    test_vals = test_df[['demand_mw']].values
    
    scaler.fit(train_vals)
    scaled_train = scaler.transform(train_vals)
    scaled_test = scaler.transform(test_vals)
    
    # Save scaler
    joblib.dump(scaler, scaler_save_path)
    
    # Sequences creation
    X_train, y_train = create_sequences(scaled_train, seq_length=seq_length)
    X_test, y_test = create_sequences(scaled_test, seq_length=seq_length)
    
    train_dataset = TimeSeriesSequenceDataset(X_train, y_train)
    test_dataset = TimeSeriesSequenceDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = PyTorchLSTMRegressor(input_size=1, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
        train_loss /= len(train_dataset)
        if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
            print(f"LSTM Epoch {epoch+1}/{epochs} - Loss: {train_loss:.6f}")
            
    # Evaluation
    model.eval()
    test_preds_scaled = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).squeeze()
            test_preds_scaled.extend(preds.cpu().numpy().tolist())
            
    test_preds_scaled = np.array(test_preds_scaled).reshape(-1, 1)
    test_preds = scaler.inverse_transform(test_preds_scaled).flatten()
    
    # Ground truth for test sequences
    y_test_actual = test_vals[seq_length:].flatten()
    
    mae = mean_absolute_error(y_test_actual, test_preds)
    rmse = np.sqrt(mean_squared_error(y_test_actual, test_preds))
    mape = mean_absolute_percentage_error(y_test_actual, test_preds) * 100
    metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    
    # Save model state dict
    torch.save(model.state_dict(), model_save_path)
    
    return model, test_preds, y_test_actual, metrics
