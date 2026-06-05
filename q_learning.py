"""
Q-learning tabular pentru mediul de tranzactionare

Mediul:
- date istorice zilnice pentru ETF-ul ales
- un episod = o parcurgere a perioadei de antrenare

Starea:
s = (ml_signal, pozitie, trend)

Actiuni:
A = {BUY, SELL, HOLD}

Recompensa:
R(s, a, s') este calculata din randamentul zilnic al portofoliului.
Daca agentul detine activul, primeste randamentul zilnic procentual.
Daca agentul este cash, recompensa este 0.
Pentru BUY si SELL se aplica o penalizare de tranzactionare.

Q-table:
Q[s, a] retine valoarea estimata pentru actiunea a in starea s.
Politica finala pi(s) alege actiunea valida cu valoarea Q maxima.
"""

import numpy as np

from config import (
    Q_ALPHA,
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
DETINE_ACTIV = 1
POZITII_POSIBILE = [CASH, DETINE_ACTIV]

NUMAR_SEMNALE_ML = 3
NUMAR_POZITII = 2
NUMAR_TRENDS = 3
NUMAR_ACTIUNI = 3

STARI_PER_SEMNAL_ML = NUMAR_POZITII * NUMAR_TRENDS
STARI_PER_POZITIE = NUMAR_TRENDS
NUMAR_STARI = NUMAR_SEMNALE_ML * NUMAR_POZITII * NUMAR_TRENDS

PROCENTE = 100


def stare_q_learning(ml_signal, pozitie, trend):
    """
    Starea agentului RL este:
    s = (ml_signal, pozitie, trend)

    ml_signal: 0 = negativ, 1 = neutru, 2 = pozitiv
    pozitie: 0 = cash, 1 = detine activ
    trend: 0 = descendent, 1 = neutru, 2 = ascendent

    Formula transforma starea compusa intr-un index din Q-table
    """
    return (
        int(ml_signal) * STARI_PER_SEMNAL_ML
        + int(pozitie) * STARI_PER_POZITIE
        + int(trend)
    )


def actiuni_valide(pozitie):
    # restrictii long-only:
    # daca agentul este cash, actiunile valide sunt BUY si HOLD
    # daca agentul detine activul, actiunile valide sunt SELL si HOLD
    if pozitie == CASH:
        return [BUY, HOLD]

    return [SELL, HOLD]


def alege_actiune_greedy(q_table, stare, actiuni_posibile):
    # daca valorile sunt egale, HOLD este alegerea neutra si evita tranzactii inutile
    actiune_aleasa = HOLD if HOLD in actiuni_posibile else actiuni_posibile[0]
    valoare_maxima = q_table[stare, actiune_aleasa]

    for actiune in actiuni_posibile:
        valoare = q_table[stare, actiune]
        if valoare > valoare_maxima:
            valoare_maxima = valoare
            actiune_aleasa = actiune

    return actiune_aleasa


def alege_actiune(q_table, stare, actiuni_posibile, epsilon, generator_random):
    if generator_random.random() < epsilon:
        return generator_random.choice(actiuni_posibile)

    return alege_actiune_greedy(q_table, stare, actiuni_posibile)


def actualizeaza_pozitia(pozitie, actiune):
    if actiune == BUY:
        return DETINE_ACTIV
    if actiune == SELL:
        return CASH
    return pozitie


def calculeaza_recompensa(
    randament_zilnic,
    pozitie_curenta,
    actiune,
    penalizare_tranzactie,
):
    # recompensa urmareste aceeasi cronologie ca benchmark-ul:
    # portofoliul curent castiga/pierde pe miscarea zilnica, apoi tranzactia are cost
    randament_piata = randament_zilnic * PROCENTE

    if pozitie_curenta == DETINE_ACTIV:
        recompensa = randament_piata
    else:
        recompensa = 0.0 # fara castig sau pierdere daca suntem cash

    if actiune in [BUY, SELL]:
        recompensa -= penalizare_tranzactie

    return recompensa


def initializeaza_q_table():
    q_table = np.zeros((NUMAR_STARI, NUMAR_ACTIUNI))
    return q_table


def train_q_learning(
    date_train,
    penalizare_tranzactie=Q_PENALIZARE_TRANZACTIE,
    seed=Q_SEED_RANDOM,
):
    q_table = initializeaza_q_table()
    generator_random = np.random.default_rng(seed)

    randuri = date_train[["ml_signal", "trend", "Close"]].to_dict("records")

    for episod in range(Q_EPISODES):
        epsilon = max(Q_EPSILON_MIN, Q_EPSILON_START * (Q_EPSILON_DECAY ** episod))

        for i in range(len(randuri) - 1):
            rand = randuri[i]
            rand_urmator = randuri[i + 1]
            randament_zilnic = rand_urmator["Close"] / rand["Close"] - 1

            # Q-table-ul are putine stari, deci putem invata ambele pozitii posibile
            for pozitie in POZITII_POSIBILE:
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
                    randament_zilnic,
                    pozitie,
                    actiune,
                    penalizare_tranzactie,
                )

                stare_urmatoare = stare_q_learning(
                    rand_urmator["ml_signal"],
                    pozitie_urmatoare,
                    rand_urmator["trend"],
                )
                actiuni_posibile_urmatoare = actiuni_valide(pozitie_urmatoare)
                max_q_urmator = max(
                    q_table[stare_urmatoare, a_urmatoare]
                    for a_urmatoare in actiuni_posibile_urmatoare
                )

                q_table[stare, actiune] = q_table[stare, actiune] + Q_ALPHA * (
                    recompensa
                    + Q_GAMMA * max_q_urmator
                    - q_table[stare, actiune]
                )

    return q_table


def alege_actiune_finala(q_table, stare, pozitie):
    actiuni_posibile = actiuni_valide(pozitie)
    return alege_actiune_greedy(q_table, stare, actiuni_posibile)
