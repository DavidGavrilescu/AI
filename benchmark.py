import numpy as np
import pandas as pd

from config import CASH_INITIAL, RANDOM_RUNS, RANDOM_SEED, TRANSACTION_COST
from q_learning import BUY as Q_BUY
from q_learning import SELL as Q_SELL
from q_learning import alege_actiune_finala, stare_q_learning


BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"


def simuleaza_actiuni(date_test, alege_actiunea):
    preturi = date_test["Close"].to_numpy()
    cash = CASH_INITIAL
    actiuni_detinute = 0
    tranzactii_facute = 0
    zile_in_piata = 0

    # decizia de azi se executa maine
    for i in range(len(preturi) - 1):
        actiune = alege_actiunea(i, actiuni_detinute)
        pret = preturi[i + 1]

        if actiune == BUY and actiuni_detinute == 0:
            actiuni_detinute = cash * (1 - TRANSACTION_COST) / pret
            cash = 0
            tranzactii_facute += 1

        elif actiune == SELL and actiuni_detinute > 0:
            cash = actiuni_detinute * pret * (1 - TRANSACTION_COST)
            actiuni_detinute = 0
            tranzactii_facute += 1

        if actiuni_detinute > 0:
            zile_in_piata += 1

    valoare_finala = cash + actiuni_detinute * preturi[-1]
    expunere = zile_in_piata / len(date_test) * 100

    return valoare_finala, tranzactii_facute, expunere


def ruleaza_buy_and_hold(date_test):
    preturi = date_test["Close"].to_numpy()
    actiuni_detinute = CASH_INITIAL * (1 - TRANSACTION_COST) / preturi[0]

    return actiuni_detinute * preturi[-1], 1, 100


def ruleaza_logistic_regression_only(date_test):
    semnale = date_test["ml_signal"].to_numpy()

    def alege_actiunea(i, actiuni_detinute):
        if semnale[i] == 2 and actiuni_detinute == 0:
            return BUY
        if semnale[i] == 0 and actiuni_detinute > 0:
            return SELL
        return HOLD

    return simuleaza_actiuni(date_test, alege_actiunea)


def ruleaza_random_agent(date_test, seed):
    generator_random = np.random.default_rng(seed)

    # daca avem actiuni putem sa vindem sau sa nu facem nimic
    # daca nu avem putem sa cumparam sau sa nu facem nimic
    def alege_actiunea(i, actiuni_detinute):
        if actiuni_detinute == 0:
            return generator_random.choice([BUY, HOLD])
        return generator_random.choice([SELL, HOLD])

    return simuleaza_actiuni(date_test, alege_actiunea)


def ruleaza_ml_q_learning(date_test, q_table):
    def alege_actiunea(i, actiuni_detinute):
        rand = date_test.iloc[i]
        pozitie = int(actiuni_detinute > 0) # 0 daca nu avem actiuni, 1 daca avem
        stare = stare_q_learning(rand["ml_signal"], pozitie, rand["trend"])
        actiune = alege_actiune_finala(q_table, stare, pozitie)

        if actiune == Q_BUY:
            return BUY
        if actiune == Q_SELL:
            return SELL
        return HOLD

    return simuleaza_actiuni(date_test, alege_actiunea)


def compara_strategii(date_test, q_table=None):
    rezultate_random = np.array([
        ruleaza_random_agent(date_test, RANDOM_SEED + i)
        for i in range(RANDOM_RUNS)
    ])

    randuri = [
        ["buy and hold", *ruleaza_buy_and_hold(date_test)],
        ["LR only", *ruleaza_logistic_regression_only(date_test)],
        ["random agent avg", *rezultate_random.mean(axis=0)],
    ]

    if q_table is not None:
        randuri.append(["ML + Q-learning", *ruleaza_ml_q_learning(date_test, q_table)])

    rezultate = pd.DataFrame(randuri, columns=[
        "strategie",
        "valoare finala",
        "tranzactii",
        "expunere %",
    ])
    rezultate["randament %"] = (rezultate["valoare finala"] / CASH_INITIAL - 1) * 100
    rezultate = rezultate[[
        "strategie",
        "valoare finala",
        "randament %",
        "tranzactii",
        "expunere %",
    ]]

    return rezultate.round({
        "valoare finala": 2,
        "randament %": 2,
        "tranzactii": 2,
        "expunere %": 2,
    })
