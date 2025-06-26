import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

trained_model = None
last_training_day = -1

def get_features(prices):
    nInst, nDays = prices.shape
    df = pd.DataFrame(prices.T)
    momentum_5 = (prices[:, 5:] - prices[:, :-5]) / prices[:, :-5]
    momentum_10 = (prices[:, 10:] - prices[:, :-10]) / prices[:, :-10]
    momentum_20 = (prices[:, 20:] - prices[:, :-20]) / prices[:, :-20]
    ma_5 = df.rolling(5).mean().values.T
    ma_20 = df.rolling(20).mean().values.T
    std_20 = df.rolling(20).std().values.T
    ma_diff = ma_5 - ma_20
    z_score = (prices - ma_20) / (std_20 + 1e-6)
    std_ratio = std_20 / (ma_20 + 1e-6)
    min_len = nDays - 20
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
    return (next_day_returns > 0).astype(int)

def train_model(prices):
    features = get_features(prices)
    labels = get_labels(prices)
    features = features[:, :-1, :]
    X = features.reshape(-1, features.shape[-1])
    y = labels.reshape(-1)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    return model

def getMyPosition(prices):
    global trained_model, last_training_day
    nInst, nDays = prices.shape
    max_dollar_position = 10000
    if nDays > 220 and (trained_model is None or nDays - last_training_day >= 10):
        trained_model = train_model(prices[:, :-1])
        last_training_day = nDays
    if trained_model is None or nDays <= 220:
        return np.zeros(nInst, dtype=int)
    features_today = get_features(prices[:, -221:])[:, -1, :]
    probs = trained_model.predict_proba(features_today)[:, 1]
    confidence = np.abs(probs - 0.5)
    direction = np.sign(probs - 0.5)
    position_value = direction * confidence * max_dollar_position
    today_prices = prices[:, -1]
    position_in_shares = (position_value / today_prices).astype(int)
    pos_limits = (max_dollar_position / today_prices).astype(int)
    position_in_shares = np.clip(position_in_shares, -pos_limits, pos_limits)
    return position_in_shares