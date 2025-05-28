'''
Student ANN Training and Validation code
'''

# Hyperparameters
num_input_units = 6  # Direct input units (no flattening)
num_output_units = 1  # Direct output units
num_hidden_layers = 4 # Number of hidden layers
num_units_per_hidden_layer = 4 # Number of units in each hidden layer
output_activation = 'linear'  # Set this to 'linear'
batch_size = 8  # Batch size for training
learning_rate = 0.01  # Learning rate
num_epochs = 2000  # Number of epochs to train

# Packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd  # For reading Excel files

# Set the random number generator
seed = 256 # Sets the random number generator seed for reproducibility
torch.manual_seed(seed)

# Color codes for terminal output
WHITE = '\033[0m'
BLUE = '\033[94m'
GREEN = '\033[92m'
LIGHT_BLUE = '\033[38;5;81m'

# File paths (assuming a single Excel file with multiple sheets for training and validating)
excel_file = 'Train Validate Apply Data.xlsx'  # Path to the Excel file containing both training and validating data
train_input_sheet = 'Train Inputs'  # Sheet name for training input patterns
train_output_sheet = 'Train Outputs'  # Sheet name for training output patterns
validate_input_sheet = 'Validate Inputs'  # Sheet name for validating input patterns
validate_output_sheet = 'Validate Outputs'  # Sheet name for validating output patterns
model_path = f'trained_ANN_{num_input_units}x{num_units_per_hidden_layer}x{num_hidden_layers}.pth'  # Path to save the trained ANN model
logFile = open('training_progress.txt', 'w')

# Write the header to the log file
logFile.write('Epoch,Train-RMSE,Validate-RMSE\n')

# Custom dataset class for Excel input/output (with sheet names and skipping the first row of titles)
class FrameDataset(Dataset):
    def __init__(self, excel_file, input_sheet, output_sheet, name='Dataset'):
        # Skip the first row (titles) by setting skiprows=1
        self.input_patterns = pd.read_excel(excel_file, sheet_name=input_sheet, header=None, skiprows=1).values
        self.output_patterns = pd.read_excel(excel_file, sheet_name=output_sheet, header=None, skiprows=1).values
        self.num_patterns = len(self.input_patterns)
        # Print number of input and output variables
        print(f'{LIGHT_BLUE}| {name} Data: Input variables = {GREEN}{self.input_patterns.shape[1]}{LIGHT_BLUE}; Output variables: {GREEN}{self.output_patterns.shape[1]} {LIGHT_BLUE}|{WHITE}')
    
    def __len__(self):
        return self.num_patterns
    
    def __getitem__(self, idx):
        input_pattern = torch.tensor(self.input_patterns[idx], dtype=torch.float32)
        output_pattern = torch.tensor(self.output_patterns[idx], dtype=torch.float32)
        return input_pattern, output_pattern

# Model definition (Fully Connected)
class FramePredictor(nn.Module):
    def __init__(self, num_input_units, num_output_units, num_hidden_layers, num_units_per_hidden_layer, output_activation='linear'):
        super(FramePredictor, self).__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(num_input_units, num_units_per_hidden_layer))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(num_units_per_hidden_layer, num_units_per_hidden_layer))
            layers.append(nn.ReLU())
        
        # Output layer (linear activation)
        layers.append(nn.Linear(num_units_per_hidden_layer, num_output_units))
        
        # Store the layers in a Sequential container
        self.ANN_layers = nn.Sequential(*layers)
        
        # No activation for the output layer (linear output)
        self.output_activation = nn.Identity()
        
    def forward(self, x):
        # No need to flatten input since it's already one-dimensional
        x = self.ANN_layers(x)
        x = self.output_activation(x)  # Apply the linear output activation
        return x  # Output remains (batch_size, num_output_units)

# Load the datasets from different sheets in the same Excel file
train_dataset = FrameDataset(excel_file, train_input_sheet, train_output_sheet, name='Training')
validate_dataset = FrameDataset(excel_file, validate_input_sheet, validate_output_sheet, name='Validating')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)
print(f'{LIGHT_BLUE}| Training patterns = {GREEN}{len(train_dataset)} {LIGHT_BLUE}|{WHITE}')
print(f'{LIGHT_BLUE}| Validating patterns = {GREEN}{len(validate_dataset)} {LIGHT_BLUE}|{WHITE}')

# Initialize the model
model = FramePredictor(num_input_units, num_output_units, num_hidden_layers, num_units_per_hidden_layer, output_activation=output_activation)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'{LIGHT_BLUE}| Using device: {GREEN}{device} {LIGHT_BLUE}|{WHITE}')
if torch.cuda.is_available():
    print(f'{LIGHT_BLUE}| GPU device found: {GREEN}{torch.cuda.get_device_name()} {LIGHT_BLUE}|{WHITE}')
model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Using Mean Squared Error loss for regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to compute RMSE from MSE
def compute_rmse(mse_loss):
    return np.sqrt(mse_loss)

# Calculate initial losses for epoch 0
def calculate_initial_losses():
    model.eval()
    with torch.no_grad():
        # Training Loss
        total_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_rmse = compute_rmse(avg_train_loss)
        
        # Validation Loss
        total_validate_loss = 0.0
        for inputs, targets in validate_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_validate_loss += loss.item()
        avg_validate_loss = total_validate_loss / len(validate_loader)
        avg_validate_rmse = compute_rmse(avg_validate_loss)
    
    return avg_train_rmse, avg_validate_rmse

# Get initial losses
initial_train_rmse, initial_validate_rmse = calculate_initial_losses()

# Print initial setup for epoch 0
print(f'{BLUE}| BEFORE TRAINING:')
print(f'{BLUE}| Train RMSE {GREEN}{initial_train_rmse:.4f}{BLUE}; Validate RMSE {GREEN}{initial_validate_rmse:.4f}{BLUE} |{WHITE}')
print(f'{BLUE}| DURING TRAINING:')

# Training loop
def train_model():
    best_train_rmse = float('inf')  # Track best train RMSE
    best_validate_rmse = float('inf')
    best_validate_epoch = 0
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        
        # Evaluate on validation data
        model.eval()
        with torch.no_grad():
            total_validate_loss = 0.0
            for inputs, targets in validate_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_validate_loss += loss.item()

            avg_validate_loss = total_validate_loss / len(validate_loader)
            
            # Convert losses to RMSE for logging
            avg_train_rmse = compute_rmse(avg_train_loss)
            avg_validate_rmse = compute_rmse(avg_validate_loss)
            
            # Check and update best losses
            if avg_validate_rmse < best_validate_rmse:
                best_validate_rmse = avg_validate_rmse
                best_validate_epoch = epoch + 1
                print(f'\r{" "*100}\r', end='')  # Clear the current line
                print(f'{BLUE}| BEST SO FAR: Epoch {GREEN}{best_validate_epoch}{BLUE}; Train RMSE {GREEN}{best_train_rmse:.4f}{BLUE}; Validate RMSE {GREEN}{best_validate_rmse:.4f}{BLUE} |')
            
            best_train_rmse = min(best_train_rmse, avg_train_rmse)  # Update with RMSE
            
            # Clear the line and print new updates
            print(f'\r{" "*100}\r', end='')  # Clear the current line
            print(f'{BLUE}| CURRENT: Epoch {GREEN}{epoch+1}/{num_epochs}{BLUE}; Train RMSE {GREEN}{avg_train_rmse:.4f}{BLUE}; Validate RMSE {GREEN}{avg_validate_rmse:.4f}{BLUE} |{WHITE}', end='')

            # Write the epoch results to the log file
            logFile.write(f'{epoch+1},{avg_train_rmse:.4f},{avg_validate_rmse:.4f}\n')
        
        model.train()

train_model()

# Save the trained model
torch.save(model.state_dict(), model_path)
logFile.close()
print(f'\n{BLUE}| Training complete: Model saved as {GREEN}{model_path}{BLUE}; Progress log file saved as {GREEN}training_progress.txt {BLUE}|{WHITE}')
