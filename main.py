import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

trained_model = None
last_training_day = -1
prev_position = None

def get_features(prices):
    nInst, nDays = prices.shape
    df = pd.DataFrame(prices.T)
    ma_5 = df.rolling(5).mean().values.T
    ma_20 = df.rolling(20).mean().values.T
    std_20 = df.rolling(20).std().values.T
    z_score = (prices - ma_20) / (std_20 + 1e-6)
    momentum_1 = (prices[:, 1:] - prices[:, :-1]) / (prices[:, :-1] + 1e-6)
    momentum_3 = (prices[:, 3:] - prices[:, :-3]) / (prices[:, :-3] + 1e-6)
    momentum_5 = (prices[:, 5:] - prices[:, :-5]) / (prices[:, :-5] + 1e-6)
    momentum_10 = (prices[:, 10:] - prices[:, :-10]) / (prices[:, :-10] + 1e-6)
    momentum_20 = (prices[:, 20:] - prices[:, :-20]) / (prices[:, :-20] + 1e-6)
    ma_diff = ma_5 - ma_20
    std_ratio = std_20 / (ma_20 + 1e-6)
    roll_min = pd.DataFrame(prices.T).rolling(10).min().values.T
    roll_max = pd.DataFrame(prices.T).rolling(10).max().values.T
    ema_10 = df.ewm(span=10, adjust=False).mean().values.T
    delta = np.diff(prices, axis=1)
    up = np.maximum(delta, 0)
    down = -np.minimum(delta, 0)
    roll_up = pd.DataFrame(up.T).rolling(14).mean().values.T
    roll_down = pd.DataFrame(down.T).rolling(14).mean().values.T
    rs = roll_up / (roll_down + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    # Additional features
    ma_50 = df.rolling(50).mean().values.T
    ema_20 = df.ewm(span=20, adjust=False).mean().values.T
    std_5 = df.rolling(5).std().values.T
    price_change_2 = (prices[:, 2:] - prices[:, :-2]) / (prices[:, :-2] + 1e-6)
    price_change_7 = (prices[:, 7:] - prices[:, :-7]) / (prices[:, :-7] + 1e-6)
    # Align new features to min_len
    min_len = prices.shape[1] - 20
    features = np.stack([
        momentum_1[:, -min_len:],
        momentum_3[:, -min_len:],
        momentum_5[:, -min_len:],
        momentum_10[:, -min_len:],
        momentum_20[:, -min_len:],
        z_score[:, -min_len:],
        ma_diff[:, -min_len:],
        std_ratio[:, -min_len:],
        roll_min[:, -min_len:],
        roll_max[:, -min_len:],
        ema_10[:, -min_len:],
        rsi[:, -min_len:],
        ma_50[:, -min_len:],
        ema_20[:, -min_len:],
        std_5[:, -min_len:],
        price_change_2[:, -(min_len):],
        price_change_7[:, -(min_len):]
    ], axis=-1)
    return features

def get_labels(prices):
    next_day_returns = (prices[:, 21:] - prices[:, 20:-1]) / (prices[:, 20:-1] + 1e-6)
    labels = (next_day_returns > 0).astype(int)
    mask = np.abs(next_day_returns) > 0.002  # Relaxed threshold for more signals
    return labels, mask

def train_model(prices):
    features = get_features(prices)
    labels, mask = get_labels(prices)
    features = features[:, :-1, :]
    X = features.reshape(-1, features.shape[-1])
    y = labels.reshape(-1)
    valid = mask.reshape(-1)
    X = X[valid]
    y = y[valid]
    # Optimized RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=300,           # more trees for stability
        max_depth=12,               # slightly deeper trees
        min_samples_leaf=5,         # avoid overfitting
        class_weight='balanced',    # handle class imbalance
        oob_score=True,             # internal validation
        random_state=2,
        n_jobs=-1
    )
    model.fit(X, y)
    # Optionally print OOB score for diagnostics
    # print(f"OOB Score: {model.oob_score_:.4f}")
    return model

def getMyPosition(prices):
    global trained_model, last_training_day, prev_position
    nInst, nDays = prices.shape
    max_dollar_position = 10000
    lookback = 300

    if nDays > lookback + 21 and (trained_model is None or nDays - last_training_day >= 5):
        train_data = prices[:, -lookback-1:-1]
        trained_model = train_model(train_data)
        last_training_day = nDays

    if trained_model is None or nDays <= lookback + 21:
        prev_position = np.zeros(nInst, dtype=int)
        return prev_position

    features_today = get_features(prices[:, -lookback:])[:, -1, :]
    probs = trained_model.predict_proba(features_today)[:, 1]
    confidence = np.abs(probs - 0.5)
    conf_threshold = 0.15
    confidence[confidence < conf_threshold] = 0
    confidence = confidence ** 2.0
    direction = np.sign(probs - 0.5)
    vol = np.std(prices[:, -20:], axis=1) / (np.mean(prices[:, -20:], axis=1) + 1e-6)
    vol[vol == 0] = 1
    position_value = 5.0 * direction * confidence * max_dollar_position / ((vol + 1e-6) ** 1.5)
    position_in_shares = (position_value / prices[:, -1]).astype(int)
    pos_limits = (max_dollar_position / prices[:, -1]).astype(int)
    raw_position = np.clip(position_in_shares, -pos_limits, pos_limits)
    if prev_position is None:
        smoothed_position = raw_position
    else:
        smoothed_position = 0.3 * raw_position + 0.7 * prev_position
    prev_position = smoothed_position.astype(int)
    return prev_position
