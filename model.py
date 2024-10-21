import torch
import torch.nn as nn
import pytorch_lightning as pl

def sharpe_ratio(returns, risk_free=0.06 / 252):
    mean_return = returns.mean()
    excess_return = mean_return - risk_free
    std_return = returns.std()
    return excess_return / (std_return + 1e-6)

class LSTMTransformerModel(pl.LightningModule):
    def __init__(self, input_dim, lstm_hidden_dim, lstm_num_layers, transformer_dim, n_heads, seq_length, dropout=0.1, transaction_cost=0.005, num_stocks=10, reg_factor=0.05, weight_volatility_factor=0.3):
        super(LSTMTransformerModel, self).__init__()
        
        self.transaction_cost = transaction_cost
        self.num_stocks = num_stocks
        self.reg_factor = reg_factor
        self.weight_volatility_factor = weight_volatility_factor

        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, lstm_num_layers, batch_first=True)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, 
            nhead=n_heads, 
            dim_feedforward=transformer_dim * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.fc = nn.Linear(transformer_dim, num_stocks)
        
        self.past_weights = None
        self.seq_length = seq_length
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        transformer_out = self.transformer(lstm_out)
        output = torch.softmax(self.fc(transformer_out[:, -1, :]), dim=1)
        return output
    
    def sharpe_loss(self, returns, portfolio_weights, past_weights):
        portfolio_return = (portfolio_weights * returns).sum(dim=1)
        sharpe = sharpe_ratio(portfolio_return)
        
        rebalancing_cost = torch.abs(portfolio_weights - past_weights).sum(dim=1) * self.transaction_cost
        weight_reg = self.reg_factor * (torch.mean(torch.var(portfolio_weights, dim=1)))
        weight_volatility_penalty = self.weight_volatility_factor * torch.mean(torch.abs(portfolio_weights - past_weights).sum(dim=1))

        return -(sharpe - rebalancing_cost.mean()) + weight_reg + weight_volatility_penalty
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        portfolio_weights = self(x)
        
        if self.past_weights is None or self.past_weights.size(0) != portfolio_weights.size(0):
            past_weights = torch.zeros_like(portfolio_weights)
        else:
            past_weights = self.past_weights.detach()

        self.past_weights = portfolio_weights.detach()
        
        loss = self.sharpe_loss(y, portfolio_weights, past_weights)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        portfolio_weights = self(x)
        loss = self.sharpe_loss(y, portfolio_weights, self.past_weights)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
