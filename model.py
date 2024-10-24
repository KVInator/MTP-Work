import torch
import torch.nn as nn
import pytorch_lightning as pl

# Sharpe Ratio calculation function
def sharpe_ratio(returns, risk_free=0.06/252):
    mean_return = returns.mean()  # Mean of portfolio returns across the batch
    excess_return = mean_return - risk_free
    std_return = returns.std()  # Standard deviation of portfolio returns across the batch
    return excess_return / (std_return + 1e-4)  # Avoid division by zero with small epsilon

class LSTMTransformerModel(pl.LightningModule):
    def __init__(self, input_dim, lstm_hidden_dim, lstm_num_layers, transformer_dim, n_heads, seq_length, dropout=0.2, transaction_cost=0.005, num_stocks=10, reg_factor=0.02):
        super(LSTMTransformerModel, self).__init__()
        
        self.transaction_cost = transaction_cost  # Transaction cost rate
        self.num_stocks = num_stocks  # Number of stocks (10 in your case)
        self.reg_factor = reg_factor  # Regularization factor for weight diversity

        # LSTM Block
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, lstm_num_layers, batch_first=True)
        
        # Transformer Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, 
            nhead=n_heads, 
            dim_feedforward=transformer_dim * 4, 
            dropout=dropout,  # Added dropout for regularization
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Fully connected layers for portfolio weights (output_dim = number of stocks)
        self.fc = nn.Linear(transformer_dim, num_stocks)  # Portfolio weights for each stock
        
        # Initialize past weights as None
        self.past_weights = None
        self.seq_length = seq_length  # Keep track of sequence length for input reshaping
    
    def forward(self, x):
        # Pass the sequence through the LSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_length, lstm_hidden_dim)
        
        # Pass through Transformer
        transformer_out = self.transformer(lstm_out)  # (batch_size, seq_length, transformer_dim)
        
        # Portfolio weights output (based on the last timestep's output of the transformer)
        output = torch.softmax(self.fc(transformer_out[:, -1, :]), dim=1)  # (batch_size, num_stocks)
        return output
    
    def sharpe_loss(self, returns, portfolio_weights, past_weights):
        """
        Calculates the Sharpe ratio loss and includes transaction costs.
        """
        portfolio_return = (portfolio_weights * returns).sum(dim=1)
        sharpe = sharpe_ratio(portfolio_return)
        
        # Transaction cost penalty for portfolio rebalancing with a threshold
        weight_diff = torch.abs(portfolio_weights - past_weights)
        rebalancing_cost = torch.where(weight_diff > 0.01, weight_diff * self.transaction_cost, torch.zeros_like(weight_diff))
        
        # Regularization term to avoid constant weights (encourages diversity in weights)
        weight_reg = self.reg_factor * (torch.mean(torch.var(portfolio_weights, dim=1)))  # Variance of portfolio weights
        
        return -(sharpe - rebalancing_cost.mean()) + weight_reg  # Add regularization term to Sharpe ratio loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch  # x is the feature sequence, y is the actual returns for the next period
    
        # Get portfolio weights from the model
        portfolio_weights = self(x)  # (batch_size, num_stocks)
    
        # Use past weights for transaction cost calculation
        if self.past_weights is None or self.past_weights.size(0) != portfolio_weights.size(0):
            past_weights = torch.zeros_like(portfolio_weights)  # Initialize for first batch
        else:
            past_weights = self.past_weights.detach()  # Detach to avoid backprop through past weights

        # Update the past weights
        self.past_weights = portfolio_weights.detach()  # Properly detach past weights
    
        # Calculate loss based on Sharpe Ratio optimization
        loss = self.sharpe_loss(y, portfolio_weights, past_weights)
    
        # Log the training loss for EarlyStopping
        self.log('train_loss', loss)
    
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.9)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}
