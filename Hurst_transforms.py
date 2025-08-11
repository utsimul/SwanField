#continue from preprocessing.py

import numpy as np
import pywt  # For wavelet transform
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

def compute_multi_scale_hurst(df_features):
    """
    Compute Hurst at multiple window sizes to identify timescales of self-similarity.
    Add this to compute_features or as a separate function after Hurst calculation.
    """
    print(BLUE + "Computing multi-scale Hurst for timescale analysis..." + ENDC)
    
    log_returns = df_features["AAPL"]["data"]['Log_Returns']
    
    # Multi-scale Hurst: Compute for different rolling windows
    windows = [10, 30, 60, 90]  # Timescales: short (10 days), medium (30-60), long (90)
    multi_hurst = {}
    
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
    
    for w in windows:
        multi_hurst[f'Hurst_{w}'] = log_returns.rolling(window=w).apply(hurst_exponent, raw=False)
    
    for key, val in multi_hurst.items():
        df_features[key] = val.interpolate().ffill().bfill()  # Impute for consistency
    
    print(f"Multi-scale Hurst columns added: {list(multi_hurst.keys())}")
    
    return df_features

def fourier_transform_analysis(log_returns):
    """
    Fourier Transform on log returns to detect dominant frequencies/timescales.
    Call this after compute_features, e.g., in process_asset_data.
    """
    print(BLUE + "Performing Fourier Transform on log returns..." + ENDC)
    
    # Drop NaNs and compute FFT
    returns_clean = log_returns.dropna().values
    N = len(returns_clean)
    yf = fft(returns_clean)
    xf = fftfreq(N, d=1)  # Daily frequency assumption (d=1 day)
    xf = xf[:N//2]  # Positive frequencies only
    power = np.abs(yf[:N//2])**2  # Power spectrum
    
    # Find peaks (dominant timescales)
    peaks, _ = find_peaks(power, height=np.mean(power) * 2)  # Threshold for significant peaks
    timescales = 1 / xf[peaks] if len(peaks) > 0 else []  # Convert frequency to period (days)
    
    print(f"Dominant timescales (days): {timescales}")
    
    # Plot Power Spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(1/xf, power)  # Plot period vs power
    plt.xlabel('Timescale (Days)')
    plt.ylabel('Power')
    plt.title('Fourier Power Spectrum: Timescales of Self-Similarity')
    plt.xscale('log')  # Log scale for better view of timescales
    plt.grid(True)
    plt.show()

def wavelet_transform_analysis(log_returns):
    """
    Wavelet Transform to localize self-similarity in time and scale.
    Call this after compute_features.
    """
    print(BLUE + "Performing Wavelet Transform on log returns..." + ENDC)
    
    returns_clean = log_returns.dropna().values
    
    # Continuous Wavelet Transform (Morlet wavelet for frequency localization)
    scales = np.arange(1, 128)  # Scales from ~1 day to ~128 days
    coef, freqs = pywt.cwt(returns_clean, scales, 'morl')  # Morlet wavelet
    
    # Plot Scalogram (wavelet power spectrum)
    plt.figure(figsize=(12, 6))
    plt.imshow(np.abs(coef), extent=[0, len(returns_clean), 1, max(scales)], cmap='PRGn', aspect='auto')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time (Days)')
    plt.ylabel('Scale (Days)')
    plt.title('Wavelet Scalogram: Localization of Self-Similarity')
    plt.yscale('log')  # Log for timescale emphasis
    plt.show()
    
    # Identify dominant scales (average power per scale)
    avg_power = np.mean(np.abs(coef), axis=1)
    dominant_scales = scales[find_peaks(avg_power)[0]]
    print(f"Dominant self-similarity scales (days): {dominant_scales}")


df_features = results.copy()
# Integration: Add to process_asset_data after df_features = compute_features(df_cleaned)
df_features = compute_multi_scale_hurst(df_features)

# Call transforms after normalization or in a separate analysis step
fourier_transform_analysis(df_features["AAPL"]["data"]['Log_Returns'])
wavelet_transform_analysis(df_features["AAPL"]["data"]['Log_Returns'])
