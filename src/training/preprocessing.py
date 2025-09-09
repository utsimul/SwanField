import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from arch import arch_model
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

# Color codes for logging
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
ENDC = '\033[0m'

def fetch_data(tickers=['AAPL', 'GOOGL'], start="2007-01-01", end="2024-12-31"):
    """Fetch OHLCV data from yfinance, preserving DatetimeIndex."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    ticker_dfs = {}
    for ticker in tickers:
        ticker_data = data.xs(ticker, axis=1, level=1) if len(tickers) > 1 else data
        ticker_data.columns = [col.lower() for col in ticker_data.columns]
        ticker_data.index = pd.to_datetime(ticker_data.index)
        ticker_dfs[ticker] = ticker_data
    print(GREEN + f"Fetched data for {tickers}" + ENDC)
    return ticker_dfs

def clean_data(df, ticker):
    """Clean data, preserving DatetimeIndex and minimizing row loss."""
    print(BLUE + f"Step 1: Cleaning data for {ticker}" + ENDC)
    if df is None or df.empty:
        print(RED + f"No data for {ticker}" + ENDC)
        return None

    print(f"Initial shape: {df.shape}, index type: {type(df.index)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"NaN counts:\n{df.isna().sum()}")

    df = df.copy()
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    print(f"After dropna OHLCV shape: {df.shape}")

    df['return'] = df['close'].pct_change().fillna(0)
    df['volatility'] = df['return'].rolling(window=21).std() * np.sqrt(252)
    df['momentum'] = df['return'].rolling(window=21).mean()

    mean_ret, std_ret = df['return'].mean(), df['return'].std()
    df = df[abs(df['return'] - mean_ret) <= 5 * std_ret]
    print(f"After outlier removal shape: {df.shape}")

    df = df.drop_duplicates()
    print(f"After drop duplicates shape: {df.shape}")

    print(BLUE + f"Cleaned data shape: {df.shape}" + ENDC)
    print(f"NaN counts after cleaning:\n{df.isna().sum()}")
    return df

def compute_features(df):
    """Compute technical indicators and features."""
    if df is None or df.empty:
        return None

    df_features = df.copy()
    close_series = df_features['close']

    df_features['SMA_10'] = SMAIndicator(close_series, window=10).sma_indicator()
    df_features['EMA_10'] = EMAIndicator(close_series, window=10).ema_indicator()
    df_features['RSI_14'] = RSIIndicator(close_series, window=14).rsi()
    bb = BollingerBands(close_series, window=10, window_dev=2)
    df_features['BB_High'] = bb.bollinger_hband()
    df_features['BB_Low'] = bb.bollinger_lband()
    df_features['Log_Returns'] = np.log(close_series / close_series.shift(1))
    df_features['Volatility_10'] = df_features['Log_Returns'].rolling(window=10).std() * np.sqrt(252)

    def hurst_exponent(series, lag=20, min_std=1e-6):
        if len(series) < lag:
            return np.nan
        series = series.dropna()
        if len(series) < 10:
            return np.nan
        lags = range(2, min(lag, len(series)))
        rs = []
        for lag in lags:
            lagged_diff = series.diff(lag).dropna()
            if len(lagged_diff) < 5:
                continue
            rs_range = lagged_diff.max() - lagged_diff.min()
            rs_std = max(lagged_diff.std(), min_std)
            rs.append(np.log(rs_range / rs_std) / np.log(lag) if rs_range > 0 else np.nan)
        return np.nanmean(rs) if rs else np.nan

    df_features['Hurst'] = df_features['Log_Returns'].rolling(window=30).apply(hurst_exponent, raw=False)

    print(BLUE + f"Features computed shape (before dropna): {df_features.shape}" + ENDC)
    print(f"NaN counts before dropna:\n{df_features.isna().sum()}")

    critical_columns = ['Log_Returns', 'Volatility_10', 'RSI_14', 'momentum', 'volume']
    df_features = df_features.loc[df_features[critical_columns].notna().all(axis=1)]
    print(f"Features computed shape (after dropna critical): {df_features.shape}")

    for col in ['SMA_10', 'EMA_10', 'BB_High', 'BB_Low', 'Hurst', 'volatility']:
        if col in df_features.columns:
            df_features[col] = df_features[col].interpolate().ffill().bfill()

    print(f"NaN counts after imputation:\n{df_features.isna().sum()}")
    return df_features

def normalize_data(df, columns=['open', 'high', 'low', 'close', 'volume', 'SMA_10', 'EMA_10', 'RSI_14', 'BB_High', 'BB_Low', 'Volatility_10']):
    """Normalize specified columns."""
    if df is None or df.empty:
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

    print(BLUE + f"Normalized data shape: {df_normalized.shape}" + ENDC)
    return df_normalized, scalers

def fit_garch_model(returns, p=1, q=1, window=200):
    """Fit a rolling GARCH model."""
    try:
        returns = returns.dropna()
        if len(returns) < 50:
            print(RED + "Insufficient data for GARCH model" + ENDC)
            return pd.Series(np.nan, index=returns.index)
        
        scale_factor = 100
        scaled_returns = returns * scale_factor
        volatility = []
        failures = 0
        for i in range(len(returns)):
            if i < window:
                volatility.append(np.nan)
                continue
            window_returns = scaled_returns.iloc[max(0, i-window):i]
            if len(window_returns) < 50:
                volatility.append(np.nan)
                continue
            try:
                model = arch_model(window_returns, vol='Garch', p=p, q=q, dist='Normal', mean='Zero', rescale=False)
                garch_fit = model.fit(disp='off', options={'maxiter': 1000})
                forecast = garch_fit.forecast(horizon=1)
                volatility.append(np.sqrt(forecast.variance.values[-1, :])[0] / scale_factor)
            except Exception as e:
                failures += 1
                volatility.append(np.nan)
        volatility_series = pd.Series(volatility, index=returns.index)
        print(f"GARCH volatility forecast (last): {volatility_series.iloc[-1]:.4f}")
        print(f"GARCH NaN count: {volatility_series.isna().sum()}")
        print(f"GARCH failures: {failures}")
        return volatility_series
    except Exception as e:
        print(RED + f"GARCH model fitting failed entirely: {e}" + ENDC)
        return pd.Series(np.nan, index=returns.index)

def process_asset_data(data_dict, ticker, augment=False):
    """Process data for a single ticker."""
    df = data_dict.get(ticker)
    if df is None or len(df) < 100:
        print(RED + f"Insufficient data for {ticker}" + ENDC)
        return None, None

    print(GREEN + f"Processing ticker: {ticker}" + ENDC)
    print(f"Initial data shape: {df.shape}, date range: {df.index.min()} to {df.index.max()}")
    for s, e in [('2008-09-01', '2009-03-15'), ('2020-02-15', '2020-03-31')]:
        mask = (df.index >= pd.to_datetime(s)) & (df.index <= pd.to_datetime(e))
        print(f"{s} to {e}: {mask.sum()} rows")

    df_cleaned = clean_data(df, ticker)
    if df_cleaned is None:
        return None, None

    df_features = compute_features(df_cleaned)
    if df_features is None:
        return None, None

    returns = df_features['Log_Returns'].dropna()
    garch_vol = fit_garch_model(returns)
    df_features['GARCH_Vol'] = garch_vol
    if df_features['GARCH_Vol'].isna().all():
        print(YELLOW + f"GARCH failed for {ticker}; using rolling std as fallback" + ENDC)
        df_features['GARCH_Vol'] = df_features['Log_Returns'].rolling(window=21).std() * np.sqrt(252)
    df_features['GARCH_Vol'] = df_features['GARCH_Vol'].interpolate().ffill().bfill()
    print(BLUE + "trying on fixes" + ENDC)
    df_features['GARCH_Vol'] = np.sqrt(df_features['GARCH_Vol']) #this works
    #df['GARCH_Vol'] = df['GARCH_Vol'] / np.sqrt(252)

    df_normalized, scalers = normalize_data(df_features)
    if df_normalized is None:
        return None, None

    autoencoder_columns = ['open', 'high', 'low', 'close', 'volume', 'SMA_10', 'EMA_10', 'RSI_14', 'BB_High', 'BB_Low', 'Volatility_10']
    for col in autoencoder_columns:
        if col in df_normalized.columns:
            min_val, max_val = df_normalized[col].min(), df_normalized[col].max()
            if not (0 <= min_val <= max_val <= 1):
                print(YELLOW + f"Warning: Column {col} not scaled properly (min={min_val:.4f}, max={max_val:.4f})" + ENDC)

    for s, e in [('2008-09-01', '2009-03-15'), ('2020-02-15', '2020-03-31')]:
        mask = (df_normalized.index >= pd.to_datetime(s)) & (df_normalized.index <= pd.to_datetime(e))
        print(f"{s} to {e} after processing: {mask.sum()} rows")
        print(f"Feature ranges in {s} to {e}:")
        for col in ['Log_Returns', 'Volatility_10', 'GARCH_Vol', 'momentum', 'RSI_14', 'volume', 'Hurst']:
            if col in df_normalized.columns:
                min_val = df_normalized[mask][col].min()
                max_val = df_normalized[mask][col].max()
                mean_val = df_normalized[mask][col].mean()
                print(f"  {col}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}")

    print(GREEN + f"Final processed data for {ticker} shape: {df_normalized.shape}" + ENDC)
    return df_normalized, scalers

def process_multiple_assets(data_dict, tickers, augment=False):
    """Process multiple tickers."""
    results = {}
    for ticker in tickers:
        df_processed, scalers = process_asset_data(data_dict, ticker, augment=augment)
        results[ticker] = {'data': df_processed, 'scalers': scalers}
        print(GREEN + f"Processed data for ticker {ticker}" + ENDC)
    return results

if __name__ == "__main__":
    data_dict = fetch_data(tickers=['AAPL', 'GOOGL'])
    results = process_multiple_assets(data_dict, ['AAPL', 'GOOGL'], augment=False)
    for ticker, result in results.items():
        df_processed = result['data']
        if df_processed is not None:
            print(f"Processed data shape for {ticker}: {df_processed.shape}")
            print(f"Date range: {df_processed.index.min()} to {df_processed.index.max()}")
            print(f"GARCH Volatility (last): {df_processed['GARCH_Vol'].iloc[-1]:.4f}")
