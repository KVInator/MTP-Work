import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from preprocess import preprocess_stock_data
from model import LSTMTransformerModel

torch.set_float32_matmul_precision('high')

def train_model(data_folder, model_folder, result_folder, reg_factor=0.05, transaction_cost=0.01, max_epochs=50):
    # Ensure the output directories exist
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)
    
    # Load CSV files from the data folder
    csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    # Preprocess the stock data, setting a sequence length of 60 for LSTM
    seq_length = 20  # Sequence length for LSTM
    X_scaled, y_scaled, dates, _ = preprocess_stock_data(csv_files, seq_length=seq_length)
    
    # Convert the preprocessed data into PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    # Create a dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8, persistent_workers=True)
    
    # Initialize the LSTM-Transformer model with regularization and transaction cost parameters
    input_dim = X_scaled.shape[2]  # Number of features per time step
    model = LSTMTransformerModel(
        input_dim=input_dim, 
        lstm_hidden_dim=64, 
        lstm_num_layers=4, 
        transformer_dim=64, 
        n_heads=8, 
        seq_length=seq_length,
        transaction_cost=transaction_cost, 
        reg_factor=reg_factor
    )
    
    # Define the device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move model to the correct device
    model = model.to(device)
    
    # Define the checkpoint callback to save the best-performing model based on training loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_folder,
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='train_loss',
        mode='min'
    )
    
    # Define an early stopping callback based on training loss
    early_stopping_callback = EarlyStopping(
        monitor='train_loss',
        patience=10,  # Reduced patience
        mode='min',
        verbose=True
    )
    
    # Use the PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator="auto"  # Automatically use GPU if available, else CPU
    )
    
    # Train the model
    trainer.fit(model, dataloader)
    
    # Load the best model checkpoint with the required parameters
    best_model_path = os.path.join(model_folder, 'best-checkpoint.ckpt')
    
    # Load the state from the checkpoint (note: call load_from_checkpoint on the class, not the instance)
    model = LSTMTransformerModel.load_from_checkpoint(
        best_model_path,
        input_dim=input_dim,  # Same input_dim used in training
        lstm_hidden_dim=64, 
        lstm_num_layers=4, 
        transformer_dim=64, 
        n_heads=8, 
        seq_length=seq_length,  # Same sequence length used in training
        transaction_cost=transaction_cost,
        reg_factor=reg_factor
    )
    
    # Move model to the correct device
    model = model.to(device)
    
    # Set the model to evaluation mode before inference
    model.eval()
    
    # Generate final portfolio weights using the trained model
    with torch.no_grad():
        final_weights = []
        for i in range(0, len(X_tensor), 32):  # Loop through data in batches
            # Move input data to the same device as the model
            batch_weights = model(X_tensor[i:i + 32].to(device)).detach().cpu().numpy()
            final_weights.append(batch_weights)

        # Stack all batches together to get the final portfolio weights
        final_weights = np.vstack(final_weights)
    
    # Save the portfolio weights along with corresponding dates
    result_path = os.path.join(result_folder, 'portfolio_weights.csv')
    
    # Flatten the dates list and match the number of weights generated
    dates_flat = np.array([d for sublist in dates for d in sublist])[:len(final_weights)]  # Flatten and align dates
    
    # Save weights and dates to CSV
    final_data = np.column_stack((dates_flat, final_weights))
    np.savetxt(result_path, final_data, delimiter=',', fmt='%s')
    
    print(f"Results saved at {result_path}")

if __name__ == "__main__":
    # Define paths
    data_folder = "data/"
    model_folder = "models/"
    result_folder = "results/"
    
    # Train the model and save results with regularization and transaction cost
    train_model(data_folder, model_folder, result_folder, reg_factor=0.2, transaction_cost=0.005, max_epochs=50)
