import numpy as np
import pandas as pd

from config import INITIAL_CASH, RANDOM_RUNS, RANDOM_SEED, TRANSACTION_COST


def simuleaza_actiuni(test_data, alege_actiunea):
    prices = test_data["Close"].to_numpy()
    cash = INITIAL_CASH
    shares = 0
    trades = 0

    # actiuni: 0 = sell, 1 = hold, 2 = buy
    # decizia de azi se executa maine
    for i in range(len(prices) - 1):
        action = alege_actiunea(i, shares)
        price = prices[i + 1]

        if action == 2 and shares == 0:
            shares = cash * (1 - TRANSACTION_COST) / price
            cash = 0
            trades += 1

        elif action == 0 and shares > 0:
            cash = shares * price * (1 - TRANSACTION_COST)
            shares = 0
            trades += 1

    final_value = cash + shares * prices[-1]
    return final_value, trades


def ruleaza_buy_and_hold(test_data):
    prices = test_data["Close"].to_numpy()
    shares = INITIAL_CASH * (1 - TRANSACTION_COST) / prices[0]

    return shares * prices[-1], 1


def ruleaza_logistic_regression_only(test_data):
    signals = test_data["ml_signal"].to_numpy()

    return simuleaza_actiuni(
        test_data,
        lambda i, shares: signals[i],
    )


def ruleaza_random_agent(test_data, seed):
    rng = np.random.default_rng(seed)

    return simuleaza_actiuni(
        test_data,
        lambda i, shares: rng.choice([2, 1] if shares == 0 else [0, 1]),
    )


def compara_benchmark(test_data):
    random_results = np.array([
        ruleaza_random_agent(test_data, RANDOM_SEED + i)
        for i in range(RANDOM_RUNS)
    ])

    rows = [
        ["Buy and Hold", *ruleaza_buy_and_hold(test_data)],
        ["Logistic Regression-only", *ruleaza_logistic_regression_only(test_data)],
        ["Random Agent avg", *random_results.mean(axis=0)],
    ]

    results = pd.DataFrame(rows, columns=["Strategy", "Final Value", "Trades"])
    results["Return %"] = (results["Final Value"] / INITIAL_CASH - 1) * 100
    results = results[["Strategy", "Final Value", "Return %", "Trades"]]

    return results.round({
        "Final Value": 2,
        "Return %": 2,
        "Trades": 2,
    })
