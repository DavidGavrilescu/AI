import numpy as np

from config import (
    Q_ALPHA,
    Q_CASH_OPPORTUNITY_WEIGHT,
    Q_EPISODES,
    Q_EPSILON_DECAY,
    Q_EPSILON_MIN,
    Q_EPSILON_START,
    Q_GAMMA,
    Q_RANDOM_SEED,
    Q_TRADE_PENALTY,
)


BUY = 0
SELL = 1
HOLD = 2

CASH = 0
SPY = 1

NUMAR_SEMNALE_ML = 3
NUMAR_POZITII = 2
NUMAR_TRENDS = 3
NUMAR_ACTIUNI = 3

STARI_PER_SEMNAL_ML = NUMAR_POZITII * NUMAR_TRENDS
STARI_PER_POZITIE = NUMAR_TRENDS
NUMAR_STARI = NUMAR_SEMNALE_ML * NUMAR_POZITII * NUMAR_TRENDS

PROCENTE = 100
REWARD_CLIP = 5.0


def get_state(ml_signal, pozitie, trend):
    # stare unica din ml_signal + pozitie + trend
    return (
        int(ml_signal) * STARI_PER_SEMNAL_ML
        + int(pozitie) * STARI_PER_POZITIE
        + int(trend)
    )


def get_valid_actions(pozitie):
    # daca suntem cash nu putem vinde, daca avem SPY nu mai cumparam
    if pozitie == CASH:
        return [BUY, HOLD]

    return [SELL, HOLD]


def alege_actiune(q_table, stare, actiuni_valide, epsilon, rng):
    if rng.random() < epsilon:
        return rng.choice(actiuni_valide)

    return actiuni_valide[np.argmax(q_table[stare, actiuni_valide])]


def actualizeaza_pozitia(pozitie, actiune):
    if actiune == BUY:
        return SPY
    if actiune == SELL:
        return CASH
    return pozitie


def calculeaza_reward(
    rand,
    pozitie_urmatoare,
    actiune,
    cash_opportunity_weight,
    trade_penalty,
):
    # reward-ul e ce se intampla pe 5 zile dupa actiune
    randament_piata = rand["future_return_5d"] * PROCENTE
    randament_piata = float(np.clip(randament_piata, -REWARD_CLIP, REWARD_CLIP))

    if pozitie_urmatoare == SPY:
        reward = randament_piata
    else:
        # cash-ul e bun cand piata scade, dar rau cand piata urca
        reward = -cash_opportunity_weight * randament_piata

    if actiune in [BUY, SELL]:
        reward -= trade_penalty

    return reward


def initializeaza_q_table():
    q_table = np.zeros((NUMAR_STARI, NUMAR_ACTIUNI))

    # valori putin optimiste, ca argmax sa nu aleaga mereu prima actiune
    for ml_signal in range(NUMAR_SEMNALE_ML):
        for trend in range(NUMAR_TRENDS):
            stare_cash = get_state(ml_signal, CASH, trend)
            stare_spy = get_state(ml_signal, SPY, trend)

            q_table[stare_cash, BUY] = 2.0
            q_table[stare_cash, HOLD] = 1.0
            q_table[stare_spy, HOLD] = 2.5
            q_table[stare_spy, SELL] = 1.0

    return q_table


def train_q_learning(
    train_data,
    cash_opportunity_weight=Q_CASH_OPPORTUNITY_WEIGHT,
    trade_penalty=Q_TRADE_PENALTY,
    seed=Q_RANDOM_SEED,
):
    q_table = initializeaza_q_table()
    rng = np.random.default_rng(seed)

    randuri = train_data[["ml_signal", "trend", "future_return_5d"]].to_dict("records")

    for episod in range(Q_EPISODES):
        epsilon = max(Q_EPSILON_MIN, Q_EPSILON_START * (Q_EPSILON_DECAY ** episod))
        pozitie = CASH

        for i in range(len(randuri) - 1):
            rand = randuri[i]
            rand_urmator = randuri[i + 1]

            stare = get_state(rand["ml_signal"], pozitie, rand["trend"])
            actiuni_valide = get_valid_actions(pozitie)
            actiune = alege_actiune(
                q_table,
                stare,
                actiuni_valide,
                epsilon,
                rng,
            )

            pozitie_urmatoare = actualizeaza_pozitia(pozitie, actiune)
            reward = calculeaza_reward(
                rand,
                pozitie_urmatoare,
                actiune,
                cash_opportunity_weight,
                trade_penalty,
            )

            stare_urmatoare = get_state(
                rand_urmator["ml_signal"],
                pozitie_urmatoare,
                rand_urmator["trend"],
            )
            actiuni_valide_urmatoare = get_valid_actions(pozitie_urmatoare)
            cel_mai_bun_next = max(
                q_table[stare_urmatoare, actiune_urmatoare]
                for actiune_urmatoare in actiuni_valide_urmatoare
            )

            q_table[stare, actiune] += Q_ALPHA * (
                reward + Q_GAMMA * cel_mai_bun_next - q_table[stare, actiune]
            )

            pozitie = pozitie_urmatoare

    return q_table


def alege_actiune_finala(q_table, stare, pozitie):
    actiuni_valide = get_valid_actions(pozitie)
    return actiuni_valide[np.argmax(q_table[stare, actiuni_valide])]
