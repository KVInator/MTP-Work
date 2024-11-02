import os
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import numpy as np
from preprocess import preprocess_stock_data
from model import LSTMTransformerModel

torch.set_float32_matmul_precision('high')

def train_model(data_folder, model_folder, result_folder, target_return=0.001, max_epochs=100):
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)
    
    csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    # Preprocess data
    seq_length = 60 
    X_scaled, y_scaled, dates, _, _ = preprocess_stock_data(csv_files, seq_length=seq_length)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    
    input_dim = X_scaled.shape[2]
    num_assets = X_scaled.shape[3]


    dataset = TensorDataset(X_tensor, y_tensor)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=10, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=10, persistent_workers=True)

    model = LSTMTransformerModel(
        input_dim=input_dim,
        num_assets=num_assets,
        lstm_hidden_dim1=64,
        lstm_hidden_dim2=32,
        transformer_dim=8,
        num_heads=2,
        seq_length=seq_length,
        init_method="small_random"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_folder,
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=20,
        mode='min',
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Trainer setup
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        accelerator="auto",
        log_every_n_steps=50,
        gradient_clip_val=2
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Load best checkpoint
    best_model_path = os.path.join(model_folder, 'best-checkpoint.ckpt')
    model = LSTMTransformerModel.load_from_checkpoint(
        best_model_path,
        input_dim=input_dim,
        num_assets=num_assets,
        lstm_hidden_dim1=64,
        lstm_hidden_dim2=32,
        transformer_dim=8,
        num_heads=2,
        seq_length=seq_length,
        init_method="small_random"
    )
    model = model.to(device)
    
    model.eval()
    all_weights = []

    with torch.no_grad():
        for i in range(len(X_tensor)):
            single_input = X_tensor[i].unsqueeze(0).to(device)
            single_weights = model(single_input).cpu().numpy()
            all_weights.append(single_weights)
            
        all_weights = np.vstack(all_weights)

    weights_path = os.path.join(result_folder, 'portfolio_weights_full.csv')
    np.savetxt(weights_path, all_weights, delimiter=',', fmt='%.6f')
    print(f"All timestamp portfolio weights saved at {weights_path}")

if __name__ == "__main__":
    data_folder = "data/"
    model_folder = "models/"
    result_folder = "results/"
    
    train_model(data_folder, model_folder, result_folder, target_return=0.001, max_epochs=70)
