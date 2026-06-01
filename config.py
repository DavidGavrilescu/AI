# date generale ale proiectului
TICKER = "SPY"
START_DATE = "2010-01-01"
END_DATE = "2026-01-01"  # yfinance nu include data de final

TRAIN_RATIO = 0.70
TREND_THRESHOLD = 0.005
LEARNING_RATE = 0.1
EPOCHS = 5000

# daca probabilitatea prezisa de model e peste acest prag
# tunci modelul prezice ca pretul va creste
THRESHOLD_PREDICTIE = 0.5

# o sa fie folosite la Logistuic regression nu includem future_return, future_return_5d sau label, pentru ca tin de viitor
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
