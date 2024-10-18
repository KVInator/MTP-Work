import torch
import torch.nn as nn
import pytorch_lightning as pl

# Sharpe Ratio calculation function
def sharpe_ratio(returns, risk_free=0):
    mean_return = returns.mean()
    excess_return = mean_return - risk_free
    std_return = returns.std()
    return excess_return / (std_return + 1e-6)  # To avoid division by zero

class LSTMTransformerModel(pl.LightningModule):
    def __init__(self, input_dim, lstm_hidden_dim, lstm_num_layers, transformer_dim, n_heads, dropout=0.1, transaction_cost=0.002):
        super(LSTMTransformerModel, self).__init__()
        
        self.transaction_cost = transaction_cost  # Set transaction cost rate
        
        # LSTM Block
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, lstm_num_layers, batch_first=True)
        
        # Transformer Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, 
            nhead=n_heads, 
            dim_feedforward=transformer_dim * 4, 
            dropout=dropout,
            batch_first=True  # Set batch_first=True for the transformer
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Fully connected layers for portfolio weights
        self.fc = nn.Linear(transformer_dim, input_dim)  # Portfolio weights for each stock (input_dim)
        
    def forward(self, x):
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Pass through Transformer
        transformer_out = self.transformer(lstm_out)
        
        # Portfolio weights output from last time step
        output = torch.softmax(self.fc(transformer_out[:, -1, :]), dim=1)  # softmax for portfolio weights
        return output
    
    def sharpe_loss(self, returns, portfolio_weights, past_weights):
        # Reshape the returns to match portfolio_weights shape
        # If returns is shape [batch_size], we need it to be [batch_size, num_stocks]
        returns = returns.unsqueeze(1).expand_as(portfolio_weights)
        
        # Calculate portfolio return: sum of weighted returns
        portfolio_return = (portfolio_weights * returns).sum(dim=1)
        
        # Sharpe Ratio with transaction cost adjustment
        sharpe = sharpe_ratio(portfolio_return)
        
        # Calculate transaction cost penalty for portfolio rebalancing
        rebalancing_cost = torch.abs(portfolio_weights - past_weights).sum(dim=1) * self.transaction_cost
        
        # Final objective is to maximize Sharpe Ratio, thus minimize -Sharpe with transaction costs
        return -(sharpe - rebalancing_cost.mean())
    
    def training_step(self, batch, batch_idx):
        x, y = batch  # y is actual returns for the next period
        
        # Get portfolio weights
        portfolio_weights = self(x)
        
        # Assume past_weights is zero (no prior portfolio); can be extended for backtesting
        past_weights = torch.zeros_like(portfolio_weights)
        
        # Calculate the loss based on Sharpe Ratio optimization
        loss = self.sharpe_loss(y, portfolio_weights, past_weights)
        
        # Log the training loss for EarlyStopping
        self.log('train_loss', loss)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
