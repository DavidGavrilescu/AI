import numpy as np

from config import (
    Q_ALPHA,
    Q_PENALIZARE_COST_OPORTUNITATE_CASH,
    Q_EPISODES,
    Q_EPSILON_DECAY,
    Q_EPSILON_MIN,
    Q_EPSILON_START,
    Q_GAMMA,
    Q_SEED_RANDOM,
    Q_PENALIZARE_TRANZACTIE,
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
LIMITA_RECOMPENSA = 5.0


def stare_q_learning(ml_signal, pozitie, trend):
    # combinam ml_signal + pozitie + trend intr-un singur numar de stare
    return (
        int(ml_signal) * STARI_PER_SEMNAL_ML
        + int(pozitie) * STARI_PER_POZITIE
        + int(trend)
    )


def actiuni_valide(pozitie):
    # daca suntem cash nu putem vinde, daca avem SPY nu mai cumparam
    if pozitie == CASH:
        return [BUY, HOLD]

    return [SELL, HOLD]


def alege_actiune(q_table, stare, actiuni_posibile, epsilon, generator_random):
    if generator_random.random() < epsilon:
        return generator_random.choice(actiuni_posibile)

    return actiuni_posibile[np.argmax(q_table[stare, actiuni_posibile])]


def actualizeaza_pozitia(pozitie, actiune):
    if actiune == BUY:
        return SPY
    if actiune == SELL:
        return CASH
    return pozitie


def calculeaza_recompensa(
    rand,
    pozitie_urmatoare,
    actiune,
    penalizare_cash,
    penalizare_tranzactie,
):
    # recompensa este randamentul pe 5 zile dupa actiune
    randament_piata = rand["future_return_5d"] * PROCENTE
    randament_piata = float(np.clip(randament_piata, -LIMITA_RECOMPENSA, LIMITA_RECOMPENSA))

    if pozitie_urmatoare == SPY:
        recompensa = randament_piata
    else:
        # cash-ul e bun cand piata scade, dar rau cand piata urca
        recompensa = -penalizare_cash * randament_piata

    if actiune in [BUY, SELL]:
        recompensa -= penalizare_tranzactie

    return recompensa


def initializeaza_q_table():
    q_table = np.zeros((NUMAR_STARI, NUMAR_ACTIUNI))

    """
    Initializeaza Q-table-ul: pentru fiecare stare si actiune memoram o valoare Q.

    Valorile nu sunt inca invatate; sunt doar valori initiale.
    Le setam usor optimist pentru actiunile valide ca agentul sa nu porneasca
    cu toate actiunile egale si sa nu aleaga mereu prima actiune prin argmax.
    """
    for ml_signal in range(NUMAR_SEMNALE_ML):
        for trend in range(NUMAR_TRENDS):
            stare_cash = stare_q_learning(ml_signal, CASH, trend)
            stare_spy = stare_q_learning(ml_signal, SPY, trend)

            q_table[stare_cash, BUY] = 2.0
            q_table[stare_cash, HOLD] = 1.0
            q_table[stare_spy, HOLD] = 2.5
            q_table[stare_spy, SELL] = 1.0

    return q_table


def train_q_learning(
    date_train,
    penalizare_cash=Q_PENALIZARE_COST_OPORTUNITATE_CASH,
    penalizare_tranzactie=Q_PENALIZARE_TRANZACTIE,
    seed=Q_SEED_RANDOM,
):
    q_table = initializeaza_q_table()
    generator_random = np.random.default_rng(seed)

    randuri = date_train[["ml_signal", "trend", "future_return_5d"]].to_dict("records")

    for episod in range(Q_EPISODES):
        epsilon = max(Q_EPSILON_MIN, Q_EPSILON_START * (Q_EPSILON_DECAY ** episod))
        pozitie = CASH

        for i in range(len(randuri) - 1):
            rand = randuri[i]
            rand_urmator = randuri[i + 1]

            stare = stare_q_learning(rand["ml_signal"], pozitie, rand["trend"])
            actiuni_posibile = actiuni_valide(pozitie)
            actiune = alege_actiune(
                q_table,
                stare,
                actiuni_posibile,
                epsilon,
                generator_random,
            )

            pozitie_urmatoare = actualizeaza_pozitia(pozitie, actiune)
            recompensa = calculeaza_recompensa(
                rand,
                pozitie_urmatoare,
                actiune,
                penalizare_cash,
                penalizare_tranzactie,
            )

            stare_urmatoare = stare_q_learning(
                rand_urmator["ml_signal"],
                pozitie_urmatoare,
                rand_urmator["trend"],
            )
            actiuni_posibile_urmatoare = actiuni_valide(pozitie_urmatoare)
            valoare_urmatoare_maxima = max(
                q_table[stare_urmatoare, actiune_urmatoare]
                for actiune_urmatoare in actiuni_posibile_urmatoare
            )

            q_table[stare, actiune] += Q_ALPHA * (
                recompensa
                + Q_GAMMA * valoare_urmatoare_maxima
                - q_table[stare, actiune]
            )

            pozitie = pozitie_urmatoare

    return q_table


def alege_actiune_finala(q_table, stare, pozitie):
    actiuni_posibile = actiuni_valide(pozitie)
    return actiuni_posibile[np.argmax(q_table[stare, actiuni_posibile])]
