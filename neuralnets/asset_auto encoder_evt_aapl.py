
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from scipy.stats import genpareto
import matplotlib.pyplot as plt

# ---------------------------
# Configuration
# ---------------------------
asset = "AAPL"
seq_len = 30
latent_dim = 16
epochs = 40
batch_size = 64
threshold_quantile = 0.95   # POT threshold u (on training scores)
extreme_prob = 1e-3         # P(score > t_extreme) target
feature_cols = ['Log_Returns','Volatility_10','GARCH_Vol','momentum','RSI_14','volume']
# OPTIONAL: add 'Hurst' if you want: feature_cols += ['Hurst']

# Weights for weighted-MSE (emphasize returns/vol)
feature_weights = {
    'Log_Returns': 3.0,
    'Volatility_10': 2.0,
    'GARCH_Vol': 2.0,
    'momentum': 1.0,
    'RSI_14': 1.0,
    'volume': 1.0
}

# ---------------------------
# 0) Get the AAPL feature DataFrame
# expects your dict: df_features["AAPL"]["data"] as a DataFrame
# ---------------------------
def prepare_dataframe(df_raw: pd.DataFrame, feature_cols):
    df = df_raw.copy()

    # Basic cleaning/imputation
    # GARCH_Vol: forward fill, then fallback to realized vol median
    if 'GARCH_Vol' in df.columns:
        df['GARCH_Vol'] = df['GARCH_Vol'].ffill()
        if 'Volatility_10' in df.columns:
            df['GARCH_Vol'] = df['GARCH_Vol'].fillna(df['Volatility_10'].median())
        else:
            df['GARCH_Vol'] = df['GARCH_Vol'].fillna(df['GARCH_Vol'].median())

    # volume: log1p
    if 'volume' in df.columns:
        df['volume'] = np.log1p(df['volume'])

    # Drop rows that still have NaNs in required cols (e.g., early indicator warmups)
    df = df.dropna(subset=feature_cols)

    return df

def make_windows(df: pd.DataFrame, cols, seq_len: int):
    """ Build (N, T, d) windows and keep end-index to map back to dates. """
    arr = df[cols].values
    X, idx = [], []
    for i in range(len(arr) - seq_len + 1):
        X.append(arr[i:i+seq_len])
        idx.append(i + seq_len - 1)
    return np.array(X), np.array(idx)

# ---------------------------
# 1) Train/test split by excluding known crash windows
# You can pass a list of blackout intervals, or derive labels otherwise.
# ---------------------------
black_swans = [
    ("2008-09-01", "2009-03-15"),
    ("2020-02-15", "2020-03-31"),
]

def label_black_swans(df_index, black_swans):
    df_index = pd.to_datetime(df_index)   # force to DatetimeIndex
    lab = pd.Series(0, index=df_index)
    for s, e in black_swans:
        s_dt = pd.to_datetime(s)
        e_dt = pd.to_datetime(e)
        # expand to cover at least 1 valid trading day
        s_dt = df_index[df_index.get_indexer([s_dt], method="bfill")][0]
        e_dt = df_index[df_index.get_indexer([e_dt], method="ffill")][0]
        lab[(df_index >= s_dt) & (df_index <= e_dt)] = 1
    return lab.values



# ---------------------------
# 2) Build the LSTM autoencoder
# ---------------------------
def build_lstm_ae(seq_len: int, n_feat: int, latent_dim: int):
    inputs = Input(shape=(seq_len, n_feat))
    # Encoder
    x = LSTM(128, activation='tanh', return_sequences=True)(inputs)
    x = LSTM(64, activation='tanh', return_sequences=False)(x)
    z = Dense(latent_dim, activation='tanh', name='latent')(x)
    # Decoder (mirror)
    x = RepeatVector(seq_len)(z)
    x = LSTM(64, activation='tanh', return_sequences=True)(x)
    x = LSTM(128, activation='tanh', return_sequences=True)(x)
    outputs = TimeDistributed(Dense(n_feat))(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

# ---------------------------
# 3) Scoring functions
# ---------------------------
def per_window_mse(X, X_hat):
    # mean over time & features
    return np.mean((X - X_hat)**2, axis=(1,2))

def per_window_weighted_mse(X, X_hat, cols, weights_dict):
    # broadcast weights over time steps
    w = np.array([weights_dict[c] for c in cols], dtype=float)  # (d,)
    w = w / w.mean()  # optional normalization
    # shape align to (1,T,d)
    W = np.broadcast_to(w, (X.shape[0], X.shape[1], len(w)))
    se = (X - X_hat)**2
    num = np.sum(W * se, axis=(1,2))
    den = np.sum(W, axis=(1,2))
    return num / den

def mahalanobis_scores(X, X_hat, cov_inv):
    # Flatten residuals per window: (N, T*d)
    R = (X - X_hat).reshape(len(X), -1)
    # s = r^T Σ^{-1} r
    return np.einsum('bi,ij,bj->b', R, cov_inv, R)

def fit_mahalanobis_cov_inv(X_train, X_train_hat):
    R = (X_train - X_train_hat).reshape(len(X_train), -1)
    # Shrinkage covariance for stability in high-dim
    lw = LedoitWolf().fit(R)
    return lw.precision_  # Σ^{-1}

# ---------------------------
# 4) EVT via POT (GPD)
# ---------------------------
def pot_threshold_gpd(train_scores, q_u=0.95, target_prob=1e-3):
    # Threshold u at q_u
    u = np.quantile(train_scores, q_u)
    exceed = train_scores[train_scores > u] - u
    if len(exceed) < 20:
        print(f"[WARN] Only {len(exceed)} exceedances above u; consider lowering q_u")
    # Fit GPD on exceedances; fix loc=0 so y >= 0
    c, loc, scale = genpareto.fit(exceed, floc=0)
    xi, beta = c, scale
    # Empirical tail prob at u
    p_u = np.mean(train_scores > u)
    # Solve for t: P(S>t) = target_prob = p_u * (1 - F_GPD(t-u))
    prob_cond = 1 - (target_prob / p_u)
    if prob_cond <= 0:
        raise ValueError("target_prob too small relative to p_u; raise target_prob or lower q_u.")
    # Inverse CDF of GPD at (1 - target_prob / p_u)
    if abs(xi) > 1e-8:
        y_q = beta * ((1 - prob_cond)**(-xi) - 1) / xi
    else:
        y_q = -beta * np.log(1 - prob_cond)
    t_extreme = u + y_q
    return u, (xi, beta), p_u, t_extreme

# ---------------------------
# 5) Main runner
# ---------------------------
def run_pipeline(df_features, asset=asset):
    df_raw = df_features[asset]['data']
    df = prepare_dataframe(df_raw, feature_cols)
    dates = pd.to_datetime(df.index)   # ensure datetime



    # Build labels for black swans (for eval / splitting)
    y_bs = label_black_swans(dates, black_swans)

    # Windows
    X_all, end_idx = make_windows(df, feature_cols, seq_len)
    end_dates = dates[end_idx]
    y_win = np.array([int(y_bs[i-seq_len+1:i+1].any()) for i in end_idx])
    print("y_win shape:", y_win.shape)
    print("Black swan windows:", np.sum(y_win==1))
    print("Normal windows:", np.sum(y_win==0))
    print("First 20 y_win values:", y_win[:20])

    # Split: train on normal windows only (no black-swan overlap)
    train_mask = (y_win == 0)
    X_train = X_all[train_mask]
    X_test  = X_all[~train_mask]
    print(f"X train shape: {X_train.shape} X test shape: {X_test.shape}")
    test_dates = end_dates[~train_mask]
    test_labels = y_win[~train_mask]

    # Scale per-feature using training set only
    n_feat = X_all.shape[2]
    scalers = []
    X_train_s = np.zeros_like(X_train, dtype=float)
    for j, col in enumerate(feature_cols):
      sc = StandardScaler()
      # print(X_train.shape) #(1204, 30, 6)
      flat = X_train[:,:,j].reshape(-1, 1)   # (N*T, 1) 
      sc.fit(flat)
      X_train_s[:,:,j] = sc.transform(flat).reshape(X_train.shape[0], X_train.shape[1])
      scalers.append(sc)


    def transform_windows(X):
        Xs = np.zeros_like(X, dtype=float)
        print("XS shape")
        print(Xs.shape)
        for j, sc in enumerate(scalers):
            flat = X[:,:,j].reshape(-1, 1)  # flatten (windows*timesteps, 1)
            scaled = sc.transform(flat)     # scale correctly
            Xs[:,:,j] = scaled.reshape(X.shape[0], X.shape[1])  # back to (n_windows, seq_len)

        return Xs

    X_test_s = transform_windows(X_test)

    # Model
    ae = build_lstm_ae(seq_len, n_feat, latent_dim)
    ae.fit(X_train_s, X_train_s, epochs=epochs, batch_size=batch_size,
           validation_split=0.1, shuffle=True, verbose=2)

    # Reconstructions
    X_train_hat = ae.predict(X_train_s, verbose=0)
    X_test_hat  = ae.predict(X_test_s,  verbose=0)

    # Scores
    train_mse = per_window_mse(X_train_s, X_train_hat)
    test_mse  = per_window_mse(X_test_s,  X_test_hat)

    train_wmse = per_window_weighted_mse(X_train_s, X_train_hat, feature_cols, feature_weights)
    test_wmse  = per_window_weighted_mse(X_test_s,  X_test_hat, feature_cols, feature_weights)

    cov_inv = fit_mahalanobis_cov_inv(X_train_s, X_train_hat)
    train_maha = mahalanobis_scores(X_train_s, X_train_hat, cov_inv)
    test_maha  = mahalanobis_scores(X_test_s,  X_test_hat,  cov_inv)

    # EVT on your chosen score (pick ONE: mse, wmse, or maha)
    
    u, (xi, beta), p_u, t_extreme = pot_threshold_gpd(train_maha, threshold_quantile, extreme_prob)
    print(f"[EVT] u={u:.6f}  xi={xi:.4f}  beta={beta:.6f}  p_u={p_u:.4f}  t_extreme={t_extreme:.6f}")

    # Flags
    flags = test_maha > t_extreme

    # Quick evaluation (precision/recall)
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    prec = precision_score(test_labels, flags) if len(np.unique(test_labels))>1 else np.nan
    rec  = recall_score(test_labels, flags)    if len(np.unique(test_labels))>1 else np.nan
    f1   = f1_score(test_labels, flags)        if len(np.unique(test_labels))>1 else np.nan
    auc  = roc_auc_score(test_labels, test_maha) if len(np.unique(test_labels))>1 else np.nan
    print(f"[Eval] Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f} AUC={auc:.3f}")

    # Plot (score on test vs thresholds)
    plt.figure(figsize=(12,5))
    plt.plot(test_dates, test_maha, label='Mahalanobis score (test)')
    plt.axhline(u, color='orange', linestyle='--', label=f'POT u (q={threshold_quantile})')
    plt.axhline(t_extreme, color='red', linestyle='--', label=f'Extreme threshold (p={extreme_prob})')
    # Shade black-swan windows for reference
    for s,e in black_swans:
        plt.axvspan(pd.to_datetime(s), pd.to_datetime(e), color='gray', alpha=0.15)
    plt.legend()
    plt.title(f"{asset} | AE residual Mahalanobis scores with EVT thresholds")
    plt.show()

    return {
        'model': ae,
        'scalers': scalers,
        'cov_inv': cov_inv,
        'scores': {
            'train_mse': train_mse, 'test_mse': test_mse,
            'train_wmse': train_wmse, 'test_wmse': test_wmse,
            'train_maha': train_maha, 'test_maha': test_maha
        },
        'thresholds': {'u': u, 't_extreme': t_extreme, 'xi': xi, 'beta': beta, 'p_u': p_u},
        'dates': {'test_dates': test_dates},
        'flags': flags.astype(int),
        'labels': test_labels
    }

# Usage example:
# results = run_pipeline(df_features)
