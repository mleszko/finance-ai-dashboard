import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class LSTMModel(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is 3D [batch, sequence, features]
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(-1)  # Add batch and feature dimensions
        elif x.dim() == 2:
            x = x.unsqueeze(-1)  # Add feature dimension
        elif x.dim() != 3:
            raise ValueError(f"Expected input to be 1D, 2D or 3D, got {x.dim()}D instead")

        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def load_data() -> pd.DataFrame:
    """Load historical price data from CSV"""
    data_path = "data/AAPL_prices.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Price data not found at {data_path}")
    return pd.read_csv(data_path)


def prepare_sequences(data: np.ndarray, sequence_length: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare sequences for LSTM training"""
    if len(data) <= sequence_length:
        raise ValueError(f"Data length ({len(data)}) must be greater than sequence_length ({sequence_length})")

    # Ensure data is 2D
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    sequences = []
    targets = []

    # Create sequences
    for i in range(len(data) - sequence_length):
        seq = data[i:(i + sequence_length)]
        target = data[i + sequence_length]
        sequences.append(seq)
        targets.append(target)

    # Convert to tensors with proper shapes
    X = torch.FloatTensor(np.array(sequences))  # Shape: [batch, seq_len, 1]
    y = torch.FloatTensor(np.array(targets))  # Shape: [batch, 1]

    return X, y


def train_model(sequence_length: int = 20, epochs: int = 100) -> None:
    """Train LSTM model and save it"""
    # Load and prepare data
    df = load_data()
    prices = df['Close'].values

    # Scale data
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))

    # Prepare sequences
    X, y = prepare_sequences(prices_scaled, sequence_length)

    # Create and train model
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)  # X is already 3D: [batch, sequence, features]
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # Save model and scaler
    torch.save(model.state_dict(), 'models/lstm_model.pt')
    joblib.dump(scaler, 'models/scaler.pkl')


def predict_next_days(n_days: int = 7, sequence_length: int = 20) -> List[float]:
    """Predict prices for the next n days"""
    # Load data and model
    df = load_data()
    prices = df['Close'].values.reshape(-1, 1)

    if len(prices) < sequence_length:
        raise ValueError(f"Not enough data points. Need at least {sequence_length} points.")

    # Load scaler and model
    scaler = joblib.load('models/scaler.pkl')
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2)
    model.load_state_dict(torch.load('models/lstm_model.pt'))
    model.eval()

    # Scale data
    prices_scaled = scaler.transform(prices)

    # Make predictions
    predictions = []
    current_sequence = prices_scaled[-sequence_length:].copy()

    for _ in range(n_days):
        # Prepare sequence tensor [batch, seq_len, features]
        sequence_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)

        # Get prediction
        with torch.no_grad():
            pred = model(sequence_tensor)

        # Get the predicted value
        pred_value = pred.numpy()[0, 0]

        # Inverse transform the prediction
        pred_original = scaler.inverse_transform([[pred_value]])[0, 0]
        predictions.append(pred_original)

        # Update sequence
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = pred_value

    return predictions


if __name__ == "__main__":
    if not os.path.exists("models/lstm_model.pt"):
        print("Training new model...")
        train_model()
    
    print("Making predictions...")
    predictions = predict_next_days(7)
    print("Next 7 days predictions:", predictions)