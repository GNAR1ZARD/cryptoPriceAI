# %%
# Dependencies

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# %%
# Pre Processing

# Load data from csv indexed by date
data = pd.read_csv('data/bitcoin.csv')
data.set_index('date', inplace=True)

# Separate features from target
features = ['price', 'total_volume', 'market_cap']
target = 'price'

# Nomalize data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[features])

# Function creates sequences out of data
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length), :-1]
        y = data[i+seq_length, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
X, y = create_sequences(data_scaled, seq_length)

# Create tensors for PyTorch
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).view(-1, 1)

# Split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=6969)

# Prepare data for batching
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# %%
# Define Long Short-Term Memory (LSTM) machine learning model
class Cryptocurrency_LSTM_model(nn.Module):
    # initialize class
    def __init__(self, input_size, hidden_layer_size=50, output_size=1):
        super(Cryptocurrency_LSTM_model, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    # define process of feeding input data trough neural net
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        lstm_out = lstm_out[:, -1, :]
        predictions = self.linear(lstm_out)
        return predictions

model = Cryptocurrency_LSTM_model(input_size=2)  # Including 'price' along with 'total_volume' and 'market_cap'

# %%
# Training Model

# define loss function
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# track start time for plotting
import time
start_time = time.time()

from IPython.display import clear_output

# Function to plot losses and show elapsed time
def plot_losses_with_time(losses, start_time):
    elapsed_time = time.time() - start_time
    clear_output(wait=True)  # Clear the existing output
    
    plt.figure(figsize=(10,5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss - Time elapsed: {elapsed_time:.2f} seconds')
    plt.legend()
    plt.show()

# set this depending on how long you want to be training model
epochs = 1000

# store losses for plotting
losses = []

print(f'Let her rip! This will take some minutes')

for i in range(epochs):
    for seq, labels in train_loader:
        optimizer.zero_grad()
        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()
    
    # adjust as needed
    if i % 10 == 0:
        # if you want to print loss updates
        # print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        losses.append(single_loss.item())

    # adjust as needed
    if i % 100 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
    
    
plot_losses_with_time(losses, start_time)
        
# %%
# Evaluating model

# check if GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

model.eval()
preds = []
true_labels = []
with torch.no_grad():
    for seq, labels in test_loader:
        seq = seq.to(device)
        y_test_pred = model(seq)
        preds.append(y_test_pred.cpu())
        true_labels.append(labels)

preds = torch.cat(preds, dim=0).numpy()
true_labels = torch.cat(true_labels, dim=0).numpy()

# print evaluated error
mse = mean_squared_error(true_labels, preds)
print(f"Test MSE: {mse}")


# %%
# Predicting future price

model.eval()

# Start predictions from last point in dataset
last_sequence = X_test[-1].unsqueeze(0)
predictions = []

# set how many days you want to predict range(days)
for _ in range(100):
    with torch.no_grad():
        prediction = model(last_sequence)
        predictions.append(prediction.item())
        feature_size = last_sequence.shape[2]
        prediction_adjusted = prediction.view(1, 1, 1).expand(-1, -1, feature_size)
        new_sequence = torch.cat((last_sequence[:, 1:, :], prediction_adjusted), dim=1)
        last_sequence = new_sequence
        
# Placeholder for predictions
placeholder = np.zeros((len(predictions), 3))
placeholder[:, 0] = predictions

# Reverse normalization to get real price
predicted_prices_transformed = scaler.inverse_transform(placeholder)
predicted_prices = predicted_prices_transformed[:, 0]

# %%
# Plot predicted prices

actual_data = data['price']

# Combining historical and predicted prices for plotting
combined_days = np.arange(-len(actual_data), len(predicted_prices))
combined_prices = np.concatenate((actual_data, predicted_prices))

# Plotting
plt.figure(figsize=(12, 7))
plt.plot(combined_days[:len(actual_data)], actual_data, color='grey', label='Historical Bitcoin Price', alpha=0.5)
plt.plot(combined_days[len(actual_data)-1:], combined_prices[len(actual_data)-1:], color='blue', label='Predicted Bitcoin Price')
plt.title('Bitcoin Price: Historical and Predicted')
plt.xlabel('Days from Now')
plt.ylabel('Price in USD')
plt.legend()
plt.grid(True)
plt.show()


# %%



