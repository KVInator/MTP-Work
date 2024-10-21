import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from preprocess import preprocess_stock_data
from model import LSTMTransformerModel

def train_model(data_folder, model_folder, result_folder, reg_factor=0.05, transaction_cost=0.005, max_epochs=50, weight_volatility_factor=0.3):
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)
    
    csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    seq_length = 60
    X_scaled, y_scaled, dates, _ = preprocess_stock_data(csv_files, seq_length=seq_length)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    model = LSTMTransformerModel(
        input_dim=X_scaled.shape[2], 
        lstm_hidden_dim=256, 
        lstm_num_layers=8, 
        transformer_dim=256, 
        n_heads=16, 
        seq_length=seq_length,
        transaction_cost=transaction_cost, 
        reg_factor=reg_factor,  
        weight_volatility_factor=weight_volatility_factor
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_folder,
        filename='best-checkpoint',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,  
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator="auto",
        gradient_clip_val=1.0,
        log_every_n_steps=50
    )
    
    trainer.fit(model, train_loader)
    
if __name__ == "__main__":
    data_folder = "data/"
    model_folder = "models/"
    result_folder = "results/"
    
    train_model(data_folder, model_folder, result_folder, reg_factor=0.02, transaction_cost=0.005, max_epochs=50, weight_volatility_factor=0.1)
