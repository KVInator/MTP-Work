import numpy as np

def monte_carlo_simulation(returns, num_stocks, num_iterations=1000, max_weight_bound=1.5, min_volatility_threshold=1e-3):
    """
    Enhanced Monte Carlo Simulation for portfolio optimization across timestamps.
    
    Args:
        returns (numpy array): The historical returns for stocks (shape: [timestamps, num_stocks]).
        num_stocks (int): The number of stocks in the portfolio.
        num_iterations (int): Number of Monte Carlo simulations per timestamp.
        max_weight_bound (float): Max allowable weight per stock, within [-max_weight_bound, max_weight_bound].
        min_volatility_threshold (float): Minimum threshold for portfolio volatility.

    Returns:
        all_optimal_weights (numpy array): Optimal weights for each timestamp.
        all_sharpe_ratios (numpy array): Best Sharpe ratios for each timestamp.
    """
    all_optimal_weights = []
    all_sharpe_ratios = []
    risk_free_rate = 0.06 / 252  # Daily risk-free rate

    for t in range(returns.shape[0]):
        best_sharpe_ratio = -np.inf
        optimal_weights = np.zeros(num_stocks)

        # Calculate covariance matrix up to the current timestamp with regularization for small samples
        if t > 0:
            current_cov_matrix = np.cov(returns[:t+1].T)
            # Add small regularization to avoid singular matrix errors
            current_cov_matrix += np.eye(num_stocks) * 1e-6
        else:
            current_cov_matrix = np.eye(num_stocks) * 1e-6  # Identity for first timestamp to avoid zero matrix

        for _ in range(num_iterations):
            # Generate random weights within bounds, then normalize
            weights = np.random.uniform(-max_weight_bound, max_weight_bound, num_stocks)
            weights /= np.sum(weights) + 1e-8  # Normalize to sum to 1 with small epsilon

            # Calculate portfolio return and volatility
            portfolio_return = np.dot(weights, returns[t])
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(current_cov_matrix, weights)))
            portfolio_volatility = max(portfolio_volatility, min_volatility_threshold)  # Avoid division by near-zero volatility

            # Calculate Sharpe ratio
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

            # Update best Sharpe ratio if this simulation is better
            if sharpe_ratio > best_sharpe_ratio:
                best_sharpe_ratio = sharpe_ratio
                optimal_weights = weights

        all_optimal_weights.append(optimal_weights)
        all_sharpe_ratios.append(best_sharpe_ratio)

    return np.array(all_optimal_weights), np.array(all_sharpe_ratios)
