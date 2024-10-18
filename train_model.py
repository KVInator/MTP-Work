import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from preprocess import preprocess_stock_data
from model import LSTMTransformerModel

def train_model(data_folder, model_folder, result_folder):
    # Ensure directories for models and results exist
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)
    
    # Get all CSV file paths from the data folder
    csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    # Preprocess data for multiple stocks
    X_scaled, y_scaled, _ = preprocess_stock_data(csv_files)
    
    # Convert to torch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)  # Adjust num_workers as needed
    
    # Initialize the model
    input_dim = X_scaled.shape[2]
    model = LSTMTransformerModel(input_dim=input_dim, lstm_hidden_dim=128, lstm_num_layers=2, transformer_dim=128, n_heads=8)
    
    # Define the checkpoint callback to save only the best model
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_folder,
        filename='best-checkpoint',
        save_top_k=1,  # Only save the top model
        verbose=True,
        monitor='train_loss',  # Monitor training loss to decide best model
        mode='min'
    )
    
    # Define early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor='train_loss',
        patience=3,  # Stop after 3 epochs of no improvement
        mode='min',
        verbose=True
    )
    
    # Initialize trainer with checkpoint and early stopping (no progress_bar_refresh_rate)
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )
    
    # Train the model
    trainer.fit(model, dataloader)
    
    # Save final predictions or portfolio weights in the results folder
    final_weights = model(X_tensor).detach().numpy()
    result_path = os.path.join(result_folder, 'portfolio_weights.csv')
    np.savetxt(result_path, final_weights, delimiter=',')
    print(f"Results saved at {result_path}")

if __name__ == "__main__":
    # Define the folders
    data_folder = "data/"
    model_folder = "models/"
    result_folder = "results/"
    
    # Train the model and save results
    train_model(data_folder, model_folder, result_folder)
