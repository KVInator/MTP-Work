import torch
import numpy as np
import os
import pandas as pd
from preprocess import preprocess_stock_data
from model import LSTMTransformerModel

def test_model(test_csv_files, model_checkpoint_path, result_filename, seq_length=200, portfolio_type="risk_neutral"):
    if not test_csv_files:
        print(f"⚠️ No test data available for {portfolio_type}. Skipping...")
        return

    os.makedirs(os.path.dirname(result_filename), exist_ok=True)

    # Preprocess test data
    X_test_scaled, _, _, _, _ = preprocess_stock_data(test_csv_files, seq_length=seq_length)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    input_dim = X_test_scaled.shape[2]
    num_assets = X_test_scaled.shape[3]

    # Load trained model with the appropriate portfolio type
    model = LSTMTransformerModel.load_from_checkpoint(
        model_checkpoint_path,
        input_dim=input_dim,
        num_assets=num_assets,
        lstm_hidden_dim1=128,
        lstm_hidden_dim2=64,
        transformer_dim=8,
        num_heads=4,
        seq_length=seq_length,
        init_method="he",
        portfolio_type=portfolio_type
    )

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device} for {portfolio_type} portfolio")
    model = model.to(device)
    model.eval()

    all_weights = []
    with torch.no_grad():
        for i in range(len(X_test_tensor)):
            single_input = X_test_tensor[i].unsqueeze(0).to(device)
            single_weights = model(single_input).cpu().numpy()
            all_weights.append(single_weights)

    all_weights = np.vstack(all_weights)

    np.savetxt(result_filename, all_weights, delimiter=',', fmt='%.6f')
    print(f"✅ Test portfolio weights for {portfolio_type} saved at {result_filename}")

if __name__ == "__main__":
    test_data_folder = "/Users/pranayvij/MTP-Work/test_data/"
    model_folder = "/Users/pranayvij/MTP-Work/models/"
    result_folder = "/Users/pranayvij/MTP-Work/results/"  # ✅ Same folder as training results

    # Define test data paths for each category
    category_files = {
        "risk_averse": os.path.join("/Users/pranayvij/MTP-Work/categorized_results/", "risk_averse_stocks.csv"),
        "risk_neutral": os.path.join("/Users/pranayvij/MTP-Work/categorized_results/", "risk_neutral_stocks.csv"),
        "risk_seeking": os.path.join("/Users/pranayvij/MTP-Work/categorized_results/", "risk_seeking_stocks.csv"),
    }

    for category, file_path in category_files.items():
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}. Skipping {category} testing.")
            continue

        # Load tickers for this category
        category_df = pd.read_csv(file_path)
        tickers = category_df['Ticker'].tolist()

        # Filter test stock CSVs for this category
        test_csv_files = [os.path.join(test_data_folder, f"{ticker}_test.csv") for ticker in tickers if os.path.exists(os.path.join(test_data_folder, f"{ticker}_test.csv"))]

        # Define model checkpoint and result file path
        model_checkpoint_path = os.path.join(model_folder, category, "best-checkpoint.ckpt")
        result_filename = os.path.join(result_folder, f"test_portfolio_weights_{category}.csv")  # ✅ Same folder as training results

        print(f"🚀 Testing model for {category.upper()} portfolio...")
        test_model(test_csv_files, model_checkpoint_path, result_filename, seq_length=60, portfolio_type=category)
