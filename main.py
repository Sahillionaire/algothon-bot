import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

trained_model = None
last_training_day = -1

def get_features(prices):
    nInst, nDays = prices.shape
    df = pd.DataFrame(prices.T)
    ma_5 = df.rolling(5).mean().values.T
    ma_20 = df.rolling(20).mean().values.T
    std_20 = df.rolling(20).std().values.T
    z_score = (prices - ma_20) / (std_20 + 1e-6)
    momentum_5 = (prices[:, 5:] - prices[:, :-5]) / (prices[:, :-5] + 1e-6)
    momentum_10 = (prices[:, 10:] - prices[:, :-10]) / (prices[:, :-10] + 1e-6)
    momentum_20 = (prices[:, 20:] - prices[:, :-20]) / (prices[:, :-20] + 1e-6)
    ma_diff = ma_5 - ma_20
    std_ratio = std_20 / (ma_20 + 1e-6)

    min_len = prices.shape[1] - 20
    features = np.stack([
        momentum_5[:, -min_len:],
        momentum_10[:, -min_len:],
        momentum_20[:, -min_len:],
        z_score[:, -min_len:],
        ma_diff[:, -min_len:],
        std_ratio[:, -min_len:]
    ], axis=-1)
    return features

def get_labels(prices):
    next_day_returns = (prices[:, 21:] - prices[:, 20:-1]) / (prices[:, 20:-1] + 1e-6)
    labels = (next_day_returns > 0).astype(int)
    mask = np.abs(next_day_returns) > 0.005
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
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    return model

def getMyPosition(prices):
    global trained_model, last_training_day
    nInst, nDays = prices.shape
    max_dollar_position = 20000
    lookback = 300

    if nDays > lookback + 21 and (trained_model is None or nDays - last_training_day >= 5):
        train_data = prices[:, -lookback-1:-1]
        trained_model = train_model(train_data)
        last_training_day = nDays

    if trained_model is None or nDays <= lookback + 21:
        return np.zeros(nInst, dtype=int)

    features_today = get_features(prices[:, -lookback:])[:, -1, :]
    probs = trained_model.predict_proba(features_today)[:, 1]
    confidence = np.abs(probs - 0.5)

    # Confidence filtering and boosting
    conf_threshold = 0.10
    confidence[confidence < conf_threshold] = 0
    confidence = confidence ** 3.0

    direction = np.sign(probs - 0.5)
    position_value = direction * confidence * max_dollar_position
    position_in_shares = (position_value / prices[:, -1]).astype(int)

    pos_limits = (max_dollar_position / prices[:, -1]).astype(int)
    return np.clip(position_in_shares, -pos_limits, pos_limits)
