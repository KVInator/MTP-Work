import torch
import torch.nn as nn
import pytorch_lightning as pl

class LSTMTransformerModel(pl.LightningModule):
    def __init__(self, input_dim=5, num_assets=10, lstm_hidden_dim1=512, lstm_hidden_dim2=256, 
                 transformer_dim=128, num_heads=4, seq_length=50, weight_penalty_factor=0.01, 
                 dropout_prob=0.2, init_method="small_random", portfolio_type="risk_neutral"):
        super(LSTMTransformerModel, self).__init__()
        self.seq_length = seq_length
        self.num_assets = num_assets
        self.weight_penalty_factor = weight_penalty_factor
        self.init_method = init_method
        self.portfolio_type = portfolio_type  # 🔹 Portfolio type determines loss function

        self.lstm1 = nn.LSTM(input_dim, lstm_hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.lstm2 = nn.LSTM(lstm_hidden_dim1, lstm_hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_prob)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden_dim2, nhead=num_heads, dim_feedforward=transformer_dim, dropout=dropout_prob, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Linear(lstm_hidden_dim2, 1)
        self._init_weights()

        self.train_losses = []
        self.val_losses = []

    def _init_weights(self):
        if self.init_method == "he":
            nn.init.kaiming_uniform_(self.fc.weight, a=0, mode='fan_in', nonlinearity='relu')
        elif self.init_method == "small_random":
            nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        else:
            nn.init.xavier_uniform_(self.fc.weight)
        
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        batch_size, seq_length, num_features, num_assets = x.shape
        all_asset_outputs = []
        
        for i in range(num_assets):
            asset_data = x[:, :, :, i]  # Shape: [batch_size, seq_length, num_features]
            asset_lstm_out, _ = self.lstm1(asset_data)
            asset_lstm_out = self.dropout1(asset_lstm_out)
            asset_lstm_out, _ = self.lstm2(asset_lstm_out)
            asset_lstm_out = self.dropout2(asset_lstm_out)
            all_asset_outputs.append(asset_lstm_out[:, -1, :])

        asset_tensor = torch.stack(all_asset_outputs, dim=1)  # Shape: [batch_size, num_assets, lstm_hidden_dim2]
        transformed_output = self.transformer(asset_tensor)

        portfolio_weights = self.fc(transformed_output).squeeze(-1)  # Shape: [batch_size, num_assets]
        portfolio_weights = torch.tanh(portfolio_weights)  # 🔹 Allows both long & short positions
        portfolio_weights = portfolio_weights / (portfolio_weights.sum(dim=1, keepdim=True))  # 🔹 Ensures sum to 1

        return portfolio_weights

    def compute_loss(self, portfolio_weights, x, y, risk_free_rate=0.04, trading_days=252):
        asset_returns = x[:, -1, 3, :]  # Close price returns
        portfolio_returns = (portfolio_weights * asset_returns).sum(dim=1)

        avg_daily_portfolio_return = portfolio_returns.mean()
        daily_excess_return = avg_daily_portfolio_return - risk_free_rate
        annualized_excess_return = daily_excess_return * (trading_days ** 0.5)

        daily_portfolio_risk = portfolio_returns.std()
        annualized_portfolio_risk = daily_portfolio_risk * (trading_days ** 0.5)

        # 🔹 **Stronger Weight Clipping Penalty**
        excess_weights = torch.abs(portfolio_weights) - 1  
        excess_penalty = torch.sum(torch.square(torch.relu(excess_weights)))  # Quadratic penalty for large deviations

        if self.portfolio_type == "risk_averse":
            # **Sortino Ratio Loss** (Penalizes only downside risk)
            downside_risk = portfolio_returns[portfolio_returns < 0].std()
            annualized_downside_risk = downside_risk * (trading_days ** 0.5)
            sortino_loss = -(annualized_excess_return / (annualized_downside_risk + 1e-4))
            total_loss = sortino_loss +  0.1*excess_penalty #+ l1_penalty

        elif self.portfolio_type == "risk_seeking":
            # **Return Maximization with Risk Constraint**
            total_loss = -annualized_excess_return +  0.1*excess_penalty #+ l1_penalty

        else:  # Default: Risk-Neutral (Sharpe Ratio Maximization)
            sharpe_loss = -(annualized_excess_return / (annualized_portfolio_risk + 1e-4))
            total_loss = sharpe_loss + 0.1*excess_penalty

        # 🔹 **Entropy Loss (Encourages Diversified Allocation)**
        entropy_loss = -torch.sum(portfolio_weights * torch.log(torch.abs(portfolio_weights) + 1e-6), dim=1).mean()
        entropy_loss = entropy_loss / torch.log(torch.tensor(portfolio_weights.shape[1], dtype=torch.float32))

        return total_loss + entropy_loss


    def training_step(self, batch, batch_idx):
        x, y = batch
        portfolio_weights = self(x)
        total_loss = self.compute_loss(portfolio_weights, x, y)
        self.train_losses.append(total_loss.detach())
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        portfolio_weights = self(x)
        val_loss = self.compute_loss(portfolio_weights, x, y)
        self.val_losses.append(val_loss.detach())
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss

    def on_train_epoch_end(self):
        avg_train_loss = torch.stack(self.train_losses).mean()
        print(f"\nEpoch {self.current_epoch}: Training Loss: {avg_train_loss:.4f}")
        self.train_losses.clear()

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.val_losses).mean()
        print(f"Epoch {self.current_epoch}: Validation Loss: {avg_val_loss:.4f}")
        self.val_losses.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold=0.001)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
