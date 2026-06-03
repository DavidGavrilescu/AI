# date generale ale proiectului
TICKER = "SPY"
START_DATE = "2010-01-01"
END_DATE = "2019-01-01" # yfinance nu include data de final
DATA_FILE = "spy_data.csv"

TRAIN_RATIO = 0.70
TREND_THRESHOLD = 0.005
LEARNING_RATE = 0.1
EPOCHS = 5000

# benchmark
CASH_INITIAL = 100000
TRANSACTION_COST = 0.0005 # 0.05%
RANDOM_RUNS = 30
RANDOM_SEED = 42

# q-learning
Q_EPISODES = 500
Q_ALPHA = 0.1
Q_GAMMA = 0.9
Q_EPSILON_START = 0.3 # explorare ridicata la inceput
Q_EPSILON_MIN = 0.01 # explorare minima la sfarsit
Q_EPSILON_DECAY = 0.993 # scade treptat explorarea
Q_PENALIZARE_TRANZACTIE = TRANSACTION_COST * 100 # cost real, pe scala reward-ului
Q_PENALIZARE_COST_OPORTUNITATE_CASH = 0.25 # cash-ul e penalizat partial cand piata urca
Q_SEED_RANDOM = 42

# prag pentru predictia LR
THRESHOLD_PREDICTIE = 0.5

# zona de incertitudine pentru ml_signal
SIGNAL_MARGIN = 0.01

# feature-uri pentru LR, fara coloane din viitor
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
