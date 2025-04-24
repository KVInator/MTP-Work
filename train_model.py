import os
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import numpy as np
import pandas as pd
from preprocess import preprocess_stock_data
from model import LSTMTransformerModel

torch.set_float32_matmul_precision('high')

def train_model(csv_files, model_folder, result_filename, max_epochs=50, portfolio_type="risk_neutral"):
    if not csv_files:
        print(f"⚠️ No data available for training in {model_folder}. Skipping...")
        return

    os.makedirs(model_folder, exist_ok=True)
    
    # Preprocess data
    seq_length = 200 
    X_scaled, y_scaled, dates, _, _ = preprocess_stock_data(csv_files, seq_length=seq_length)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    input_dim = X_scaled.shape[2]
    num_assets = X_scaled.shape[3]

    dataset = TensorDataset(X_tensor, y_tensor)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=6, persistent_workers=True)

    model = LSTMTransformerModel(
        input_dim=input_dim,
        num_assets=num_assets,
        lstm_hidden_dim1=128,
        lstm_hidden_dim2=64,
        transformer_dim=8,
        num_heads=4,
        seq_length=seq_length,
        init_method="he",
        portfolio_type=portfolio_type  # 🔹 Pass portfolio type here
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
        patience=15,
        mode='min',
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Trainer setup
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        accelerator="auto",
        log_every_n_steps=50
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Load best checkpoint
    best_model_path = os.path.join(model_folder, 'best-checkpoint.ckpt')
    model = LSTMTransformerModel.load_from_checkpoint(
        best_model_path,
        input_dim=input_dim,
        num_assets=num_assets,
        lstm_hidden_dim1=128,
        lstm_hidden_dim2=64,
        transformer_dim=8,
        num_heads=4,
        seq_length=seq_length,
        init_method="he",
        portfolio_type=portfolio_type  # 🔹 Ensure it's passed when reloading the model
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

    weights_path = os.path.join("results", result_filename)
    np.savetxt(weights_path, all_weights, delimiter=',', fmt='%.6f')
    print(f"✅ Portfolio weights saved at {weights_path}")

if __name__ == "__main__":
    categorized_results_folder = "/Users/pranayvij/MTP-Work/categorized_results/"
    data_folder = "/Users/pranayvij/MTP-Work/data/"

    # Define file paths for each category
    category_files = {
        "risk_averse": os.path.join(categorized_results_folder, "risk_averse_stocks.csv"),
        "risk_neutral": os.path.join(categorized_results_folder, "risk_neutral_stocks.csv"),
        "risk_seeking": os.path.join(categorized_results_folder, "risk_seeking_stocks.csv"),
    }

    for category, file_path in category_files.items():
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}. Skipping {category} training.")
            continue

        # Load tickers for this category
        category_df = pd.read_csv(file_path)
        tickers = category_df['Ticker'].tolist()

        # Filter stock CSVs for this category
        csv_files = [os.path.join(data_folder, f"{ticker}.csv") for ticker in tickers if os.path.exists(os.path.join(data_folder, f"{ticker}.csv"))]

        # Define model folder and results file
        model_folder = f"models/{category}/"
        result_filename = f"portfolio_weights_{category}.csv"

        print(f"🚀 Training model for {category.upper()} portfolio...")
        train_model(csv_files, model_folder, result_filename, max_epochs=70, portfolio_type=category)
