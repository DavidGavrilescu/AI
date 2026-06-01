# date generale ale proiectului
TICKER = "SPY"
START_DATE = "2010-01-01"
END_DATE = "2026-01-01"  # yfinance nu include data de final

TRAIN_RATIO = 0.70
TREND_THRESHOLD = 0.005
LEARNING_RATE = 0.1
EPOCHS = 5000

# benchmarks
INITIAL_CASH = 100000
TRANSACTION_COST = 0.001  # 0.1% per trade
RANDOM_RUNS = 30
RANDOM_SEED = 42

# daca probabilitatea prezisa e peste acest prag, modelul prezice clasa 1
THRESHOLD_PREDICTIE = 0.5

# zona de incertitudine pentru ml_signal
SIGNAL_MARGIN = 0.01

# folosite la Logistic Regression, fara coloane din viitor
FEATURE_COLUMNS = [
    "daily_return",
    "open_close_return",
    "high_low_range",
    "volume_change",
    "ma_ratio",
]

NORMALIZED_FEATURE_COLUMNS = [
    "daily_return_norm",
    "open_close_return_norm",
    "high_low_range_norm",
    "volume_change_norm",
    "ma_ratio_norm",
]
