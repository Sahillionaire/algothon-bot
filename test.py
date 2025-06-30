import numpy as np

prev_positions = np.zeros(50, dtype=int)

def getMyPosition(prices: np.ndarray) -> np.ndarray:
    global prev_positions
    nInst, nDays = prices.shape
    positions = np.zeros(nInst, dtype=int)

    if nDays < 200:
        return prev_positions.copy()

    signal_scores = np.zeros(nInst)

    for i in range(nInst):
        p = prices[i]
        price = p[-1]

        if price == 0:
            continue

        ma50 = np.mean(p[-50:])
        ma200 = np.mean(p[-200:])
        zscore = (price - ma200) / (np.std(p[-200:]) + 1e-6)
        momentum = (price - p[-21]) / (p[-21] + 1e-6)

        score = 0
        if ma50 > ma200: score += 1
        if momentum > 0: score += 1
        if abs(zscore) < 1: score += 1  # avoid overbought/oversold

        signal_scores[i] = score

    # Trade only top N strongest signals
    N = 10
    top_indices = np.argsort(-signal_scores)[:N]

    for i in top_indices:
        p = prices[i]
        price = p[-1]
        dollar_limit = 10000
        max_shares = int(dollar_limit // price)

        score = signal_scores[i]

        if score == 3:
            positions[i] = max_shares  # confident long
        elif score == 0:
            positions[i] = -max_shares  # confident short
        elif score == 2:
            positions[i] = int(0.5 * max_shares)
        elif score == 1:
            positions[i] = int(0.25 * max_shares)
        else:
            positions[i] = 0

    prev_positions = positions.copy()
    return positions
