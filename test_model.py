import torch
import numpy as np
import os
from preprocess import preprocess_stock_data
from model import LSTMTransformerModel

def test_model(test_data_folder, model_checkpoint_path, result_folder, seq_length=120):
    os.makedirs(result_folder, exist_ok=True)
    
    test_csv_files = [os.path.join(test_data_folder, f) for f in os.listdir(test_data_folder) if f.endswith('.csv')]
    X_test_scaled, _, _, scaler_close, scaler_volume = preprocess_stock_data(test_csv_files, seq_length=seq_length)
    
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    
    input_dim = X_test_scaled.shape[2]
    num_assets = X_test_scaled.shape[3]
    model = LSTMTransformerModel.load_from_checkpoint(
        model_checkpoint_path, 
        input_dim=input_dim, 
        num_assets=num_assets,
        lstm_hidden_dim1=128, 
        lstm_hidden_dim2=64, 
        transformer_dim=32, 
        num_heads=2, 
        seq_length=seq_length,
        init_method="he"
    )
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    
    all_weights = []
    with torch.no_grad():
        for i in range(len(X_test_tensor)):
            single_input = X_test_tensor[i].unsqueeze(0).to(device) 
            single_weights = model(single_input).cpu().numpy()
            all_weights.append(single_weights)
    
    all_weights = np.vstack(all_weights)
    weights_path = os.path.join(result_folder, 'test_portfolio_weights.csv')
    np.savetxt(weights_path, all_weights, delimiter=',', fmt='%.6f')
    print(f"Test portfolio weights saved at {weights_path}")

if __name__ == "__main__":
    test_data_folder = "test_data/"
    model_checkpoint_path = "models/best-checkpoint.ckpt"
    result_folder = "results/"
    
    test_model(test_data_folder, model_checkpoint_path, result_folder)
