import numpy as np
import pandas as pd

from config import INITIAL_CASH, RANDOM_RUNS, RANDOM_SEED, TRANSACTION_COST
from q_learning import BUY as Q_BUY
from q_learning import SELL as Q_SELL
from q_learning import alege_actiune_finala, get_state


BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"


def simuleaza_actiuni(date_test, alege_actiunea):
    preturi = date_test["Close"].to_numpy()
    cash = INITIAL_CASH
    actiuni_spy = 0
    tranzactii = 0
    zile_in_piata = 0

    # decizia de azi se executa maine
    for i in range(len(preturi) - 1):
        actiune = alege_actiunea(i, actiuni_spy)
        pret = preturi[i + 1]

        if actiune == BUY and actiuni_spy == 0:
            actiuni_spy = cash * (1 - TRANSACTION_COST) / pret
            cash = 0
            tranzactii += 1

        elif actiune == SELL and actiuni_spy > 0:
            cash = actiuni_spy * pret * (1 - TRANSACTION_COST)
            actiuni_spy = 0
            tranzactii += 1

        if actiuni_spy > 0:
            zile_in_piata += 1

    valoare_finala = cash + actiuni_spy * preturi[-1]
    expunere = zile_in_piata / len(date_test) * 100

    return valoare_finala, tranzactii, expunere


def ruleaza_buy_and_hold(date_test):
    preturi = date_test["Close"].to_numpy()
    actiuni_spy = INITIAL_CASH * (1 - TRANSACTION_COST) / preturi[0]

    return actiuni_spy * preturi[-1], 1, 100


def ruleaza_logistic_regression_only(date_test):
    semnale = date_test["ml_signal"].to_numpy()

    def alege_actiunea(i, actiuni_spy):
        if semnale[i] == 2:
            return BUY
        if semnale[i] == 0:
            return SELL
        return HOLD

    return simuleaza_actiuni(date_test, alege_actiunea)


def ruleaza_random_agent(date_test, seed):
    rng = np.random.default_rng(seed)

    def alege_actiunea(i, actiuni_spy):
        if actiuni_spy == 0:
            return rng.choice([BUY, HOLD])
        return rng.choice([SELL, HOLD])

    return simuleaza_actiuni(date_test, alege_actiunea)


def ruleaza_ml_q_learning(date_test, q_table):
    def alege_actiunea(i, actiuni_spy):
        rand = date_test.iloc[i]
        pozitie = int(actiuni_spy > 0)
        stare = get_state(rand["ml_signal"], pozitie, rand["trend"])
        actiune = alege_actiune_finala(q_table, stare, pozitie)

        if actiune == Q_BUY:
            return BUY
        if actiune == Q_SELL:
            return SELL
        return HOLD

    return simuleaza_actiuni(date_test, alege_actiunea)


def compara_benchmark(date_test, q_table=None):
    random_results = np.array([
        ruleaza_random_agent(date_test, RANDOM_SEED + i)
        for i in range(RANDOM_RUNS)
    ])

    rows = [
        ["buy and hold", *ruleaza_buy_and_hold(date_test)],
        ["LR only", *ruleaza_logistic_regression_only(date_test)],
        ["random agent avg", *random_results.mean(axis=0)],
    ]

    if q_table is not None:
        rows.append(["ML + Q-learning", *ruleaza_ml_q_learning(date_test, q_table)])

    results = pd.DataFrame(rows, columns=[
        "strategie",
        "valoare finala",
        "tranzactii",
        "expunere %",
    ])
    results["randament %"] = (results["valoare finala"] / INITIAL_CASH - 1) * 100
    results = results[[
        "strategie",
        "valoare finala",
        "randament %",
        "tranzactii",
        "expunere %",
    ]]

    return results.round({
        "valoare finala": 2,
        "randament %": 2,
        "tranzactii": 2,
        "expunere %": 2,
    })
