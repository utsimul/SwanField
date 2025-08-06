
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from arch import arch_model
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
import datetime
from scipy.stats import norm
from hmmlearn.hmm import GaussianHMM

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
ENDC = '\033[0m'

def fetch_data():
    tickers = ['AAPL','MSFT','GOOGL']
    data = yf.download(tickers, start="2020-01-01", end="2024-12-31")
    ticker_dfs = {}

    for ticker in tickers:
        ticker_data = data.xs(ticker, axis=1, level=1)
    
    
        ticker_data.columns = [col.lower() for col in ticker_data.columns]
    
    
        ticker_dfs[ticker] = ticker_data

        rolling_window=21


    # Prepare an output dictionary for enriched features
    enriched_ticker_data = {}
    
    # Iterate through each ticker
    for ticker, df in ticker_dfs.items():
        df = df.copy()
    
        # Calculate returns
        df["return"] = np.log(df["close"] / df["close"].shift(1))
    
        # Rolling 21-day volatility
        df["volatility"] = df["return"].rolling(window=rolling_window).std()
    
        # Momentum: rolling average return (21d)
        df["momentum"] = df["return"].rolling(window=21).mean()
    
        # Sharpe ratio (annualized): mean / std * sqrt(252)
        df["sharpe_ratio"] = (df["momentum"] / df["volatility"]) * np.sqrt(252)
    
        hmm_features = df[["return", "volatility"]].dropna()
    
        try:
            hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
            hmm.fit(hmm_features + 1e-6 * np.random.randn(*hmm_features.shape))  # small noise
            regimes = hmm.predict(hmm_features)
    
            # Align regimes safely
            df.loc[hmm_features.index, "regime_state"] = regimes
    
        except Exception as e:
            print(f"⚠️ HMM failed for {ticker}: {e}")
            df["regime_state"] = np.nan
    
        enriched_ticker_data[ticker] = df
    
    
    # ✅ Example: view features for AAPL
    print(enriched_ticker_data["AAPL"].head())
    return enriched_ticker_data



def clean_data(df, ticker):
    """
    Step 1: Clean data by handling missing values, outliers, and duplicates.
    """
    print(BLUE + """
    Step 1: Clean data by handling missing values, outliers, and duplicates.
    Outliers (>5σ returns) are removed to prevent data errors from skewing MARL training.
    """ + ENDC)
    if df is None:
        return None

    df = df.reset_index(drop=True)
    print(f"Step 1 - After index reset shape: {df.shape}")

    print(f"Step 1 - NaN counts before cleaning:\n{df.isna().sum()}")

    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    print(f"Step 1 - After dropna OHLCV shape: {df.shape}")

    if 'return' in df.columns:
        df['return'] = df['return'].fillna(df['close'].pct_change())
    else:
        df['return'] = df['close'].pct_change()

    mean_ret, std_ret = df['return'].mean(), df['return'].std()
    df = df[abs(df['return'] - mean_ret) <= 5 * std_ret]
    print(f"Step 1 - After outlier removal shape: {df.shape}")

    df = df.drop_duplicates()
    print(f"Step 1 - After drop duplicates shape: {df.shape}")

    df = df.drop(columns='return', errors='ignore')

    # Impute regime_state for AAPL; exclude for GOOGL if entirely NaN
    if ticker != 'GOOGL' and 'regime_state' in df.columns:
        df['regime_state'] = df['regime_state'].ffill().bfill()
    elif ticker == 'GOOGL' and 'regime_state' in df.columns and df['regime_state'].isna().all():
        df = df.drop(columns='regime_state', errors='ignore')

    print(BLUE + f"Step 1 - Cleaned data shape: {df.shape}" + ENDC)
    print(f"Step 1 - NaN counts after cleaning:\n{df.isna().sum()}")
    print(df.head())

    return df

def compute_features(df):
    """
    Step 2: Compute technical indicators and features on unnormalized data.
    """
    if df is None:
        return None

    df_features = df.copy()
    close_series = df_features['close']

    # Use shorter windows to reduce NaNs
    df_features['SMA_10'] = SMAIndicator(close_series, window=10).sma_indicator()
    df_features['EMA_10'] = EMAIndicator(close_series, window=10).ema_indicator()
    df_features['RSI_14'] = RSIIndicator(close_series, window=14).rsi()
    bb = BollingerBands(close_series, window=10, window_dev=2)
    df_features['BB_High'] = bb.bollinger_hband()
    df_features['BB_Low'] = bb.bollinger_lband()
    df_features['Log_Returns'] = np.log(close_series / close_series.shift(1))
    df_features['Volatility_10'] = df_features['Log_Returns'].rolling(window=10).std() * np.sqrt(252)

    def hurst_exponent(series, lag=20):
        if len(series) < lag:
            return np.nan
        lags = range(2, lag)
        rs = []
        for lag in lags:
            lagged_diff = series.diff(lag).dropna()
            if len(lagged_diff) == 0:
                return np.nan
            rs_range = lagged_diff.max() - lagged_diff.min()
            rs_std = lagged_diff.std()
            rs.append(np.log(rs_range / rs_std) / np.log(lag) if rs_std != 0 else 0)
        return np.mean(rs) if rs else np.nan

    df_features['Hurst'] = df_features['Log_Returns'].rolling(window=30).apply(hurst_exponent, raw=False)

    # Validate features against input volatility if available
    if 'volatility' in df_features.columns:
        correlation = df_features['Volatility_10'].corr(df_features['volatility'])
        print(f"Step 2 - Correlation between Volatility_10 and input volatility: {correlation:.4f}")

    print(BLUE + f"Step 2 - Features computed shape (before dropna): {df_features.shape}" + ENDC)
    print(f"Step 2 - NaN counts before dropna:\n{df_features.isna().sum()}")

    critical_columns = ['open', 'high', 'low', 'close', 'volume', 'Log_Returns']
    df_features = df_features.dropna(subset=critical_columns)
    print(BLUE + f"Step 2 - Features computed shape (after dropna critical): {df_features.shape}" + ENDC)

    for col in ['SMA_10', 'EMA_10', 'RSI_14', 'BB_High', 'BB_Low', 'Volatility_10', 'Hurst']:
        if col in df_features.columns:
            df_features[col] = df_features[col].interpolate().ffill().bfill()

    print(f"Step 2 - NaN counts after imputation:\n{df_features.isna().sum()}")
    print(df_features.head())

    return df_features

def normalize_data(df, columns=['open', 'high', 'low', 'close', 'volume', 'SMA_10', 'EMA_10', 'RSI_14', 'BB_High', 'BB_Low', 'Volatility_10', 'Hurst']):
    """
    Step 3: Normalize specified columns using MinMaxScaler.
    """
    if df is None:
        return None, None

    df_normalized = df.copy()
    scalers = {}

    for col in columns:
        if col in df_normalized.columns:
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(df_normalized[[col]].values.reshape(-1, 1)).flatten()
            df_normalized[col] = scaled_values
            scalers[col] = scaler
        else:
            print(YELLOW + f"Warning: Column {col} not found in DataFrame" + ENDC)

    print(BLUE + f"Step 3 - Normalized data shape: {df_normalized.shape}" + ENDC)
    print(df_normalized.head())

    return df_normalized, scalers

def augment_data(df, noise_factor=0.01, num_synthetic=100):
    """
    Step 4: Augment data with synthetic samples using Gaussian noise.
    """
    if df is None:
        return None

    synthetic_data = []
    for _ in range(num_synthetic):
        noise = norm.rvs(scale=noise_factor, size=len(df))
        synthetic_df = df.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in synthetic_df.columns:
                synthetic_df[col] = synthetic_df[col] * (1 + noise * (0.5 if col == 'volume' else 1))
        synthetic_data.append(synthetic_df)

    augmented_df = pd.concat([df] + synthetic_data, ignore_index=True)
    print(BLUE + f"Step 4 - Augmented data shape: {augmented_df.shape}" + ENDC)
    print(augmented_df.head())

    return augmented_df

def fit_garch_model(returns, p=1, q=1, window=100):
    """
    Step 5: Fit a rolling GARCH(p, q) model to forecast volatility.
    """
    try:
        returns = returns.dropna()
        if len(returns) < 50:
            print(RED + "Step 5 - Insufficient data for GARCH model" + ENDC)
            return pd.Series(np.nan, index=returns.index)
        
        # Rolling GARCH
        volatility = []
        for i in range(len(returns)):
            if i < window:
                volatility.append(np.nan)
                continue
            window_returns = returns.iloc[max(0, i-window):i]
            if len(window_returns) < 50:
                volatility.append(np.nan)
                continue
            model = arch_model(window_returns, vol='Garch', p=p, q=q, dist='Normal')
            garch_fit = model.fit(disp='off')
            forecast = garch_fit.forecast(horizon=1)
            volatility.append(np.sqrt(forecast.variance.values[-1, :])[0])
        
        volatility_series = pd.Series(volatility, index=returns.index)
        print(f"Step 5 - GARCH volatility forecast (last): {volatility_series.iloc[-1]:.4f}")
        return volatility_series
    except Exception as e:
        print(RED + f"Step 5 - GARCH model fitting failed: {e}" + ENDC)
        return pd.Series(np.nan, index=returns.index)

def process_asset_data(data_dict, ticker, augment=False):
    """
    Main function to process data for a single ticker.
    """
    df = data_dict.get(ticker)
    if df is None or len(df) < 100:
        print(RED + f"Insufficient data for {ticker}" + ENDC)
        return None, None

    print(GREEN + f"Processing ticker: {ticker}" + ENDC)
    print(f"Initial data shape: {df.shape}")
    print(f"Initial columns: {df.columns.tolist()}")

    df_cleaned = clean_data(df, ticker)
    if df_cleaned is None:
        return None, None

    df_features = compute_features(df_cleaned)
    if df_features is None:
        return None, None

    df_normalized, scalers = normalize_data(df_features)
    if df_normalized is None:
        return None, None

    if augment:
        df_normalized = augment_data(df_normalized)
        if df_normalized is None:
            return None, None

    returns = df_normalized['Log_Returns'].dropna() * 100
    garch_vol = fit_garch_model(returns)
    df_normalized['GARCH_Vol'] = garch_vol

    print(GREEN + f"Final processed data for {ticker} ------------------------------------" + ENDC)
    print(df_normalized)
    print(f"Scalers for {ticker} ------------------------------------")
    print(scalers)

    return df_normalized, scalers

def process_multiple_assets(data_dict, tickers, augment=False):
    """
    Process multiple tickers and return a dictionary of processed DataFrames and scalers.
    """
    results = {}
    for ticker in tickers:
        df_processed, scalers = process_asset_data(data_dict, ticker, augment=augment)
        results[ticker] = {'data': df_processed, 'scalers': scalers}
        print(GREEN + f"Processed data for ticker {ticker}" + ENDC)
    return results

if __name__ == "__main__":
    # Replace with your actual data dictionary
    data_dict = fetch_data()
    tickers = ['AAPL', 'GOOGL']

    results = process_multiple_assets(data_dict, tickers, augment=False)

    for ticker, result in results.items():
        df_processed = result['data']
        scalers = result['scalers']
        if df_processed is not None:
            print(f"Processed data shape for {ticker}: {df_processed.shape}")
            print(df_processed.head())
            print(f"GARCH Volatility Forecast for {ticker} (last): {df_processed['GARCH_Vol'].iloc[-1]:.4f}")
