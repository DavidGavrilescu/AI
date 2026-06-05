# proiect ai

Proiectul compara cateva strategii simple de trading pe ETF-urile SPY si QQQ.

Ideea pe scurt:
- iau date istorice din fisierele locale `SPY.csv` si `QQQ.csv`
- construiesc cateva feature-uri din pret si volum
- antrenez un Logistic Regression facut manual cu `numpy`
- transform probabilitatea modelului intr-un semnal discret
- folosesc semnalul asta intr-un agent Q-learning
- compar strategiile pe perioada de test

## Date

Datele sunt luate din fisiere CSV locale, ca rezultatele sa fie reproductibile.
Sursa datelor este Kaggle - `S&P 500 and NASDAQ 100 Daily Data`.
Fisierele folosite in proiect sunt `SPY.csv` si `QQQ.csv`.

In `config.py` se seteaza tickerul folosit de `main.py`:

```python
TICKER = "SPY"
```

Tickere disponibile:
- `SPY`
- `QQQ`

Perioada folosita este:
- start: `2010-01-01`
- final: `2019-01-01` (data de final nu este inclusa)

## Pipeline

Pipeline-ul principal face urmatorii pasi:

1. incarca datele pentru tickerul ales
2. ajusteaza preturile folosind `adjusted_close`
3. calculeaza feature-urile
4. imparte datele in train si test
5. construieste label-ul pentru Logistic Regression
6. normalizeaza feature-urile cu min/max din train
7. antreneaza Logistic Regression
8. transforma probabilitatile in `ml_signal`
9. antreneaza Q-learning
10. ruleaza benchmark-ul pe datele de test

Split-ul este cronologic:
- 70% train
- 30% test

## Feature-uri

Feature-urile folosite de Logistic Regression sunt:

- `daily_return`: randamentul fata de ziua precedenta
- `open_close_return`: miscarea dintre Open si Close
- `high_low_range`: intervalul High-Low raportat la Close
- `volume_change`: schimbarea volumului fata de ziua precedenta
- `ma_ratio`: diferenta relativa dintre media mobila pe 5 zile si cea pe 20 de zile

Mai sunt calculate:
- `trend`: 0 = descendent, 1 = neutru, 2 = ascendent
- `future_return_5d`: randamentul peste 5 zile

`future_return_5d` este folosit doar pentru target, nu ca feature de intrare.

## Logistic Regression

Logistic Regression este implementat manual in `logistic_regression.py`.

Modelul:
- foloseste sigmoid
- porneste cu ponderile 0
- se antreneaza cu gradient descent
- prezice daca randamentul viitor pe 5 zile este peste mediana din train

Pragul de predictie este `0.5`.

Dupa predictie, probabilitatea este transformata in `ml_signal`:

- `0`: probabilitate sub `0.49`
- `1`: probabilitate intre `0.49` si `0.51`
- `2`: probabilitate peste `0.51`

Zona asta de incertitudine este data de `SIGNAL_MARGIN = 0.01`.

## Q-learning

Agentul Q-learning foloseste o stare formata din:

- `ml_signal`
- pozitia curenta: cash sau investit
- `trend`

Actiunile posibile sunt:

- `BUY`
- `SELL`
- `HOLD`

Daca agentul este cash, poate alege doar `BUY` sau `HOLD`.
Daca agentul este investit, poate alege doar `SELL` sau `HOLD`.

Recompensa vine din randamentul zilnic al pozitiei curente, pe aceeasi logica folosita in benchmark. Tranzactiile au cost, iar cash-ul are randament 0.

Am doua variante:

- `ML + Q-learning`: foloseste `ml_signal`, pozitia si trendul
- `Q-learning fara ML`: ignora semnalul ML si foloseste doar pozitia si trendul

## Strategii comparate

Benchmark-ul compara:

- `buy and hold`
- `LR only`
- `random agent avg`
- `ML + Q-learning`
- `Q-learning fara ML`

Portofoliul porneste cu:

```python
CASH_INITIAL = 100000
```

Costul de tranzactie este:

```python
TRANSACTION_COST = 0.0005
```

Adica 0.05%.

In benchmark se afiseaza:
- valoare finala
- randament %
- numar de tranzactii
- expunere %
- Sharpe Ratio
- Max Drawdown %

## Fisiere

- `main.py`: ruleaza pipeline-ul principal pentru tickerul din `config.py`
- `config.py`: parametrii proiectului
- `data_processing.py`: incarca datele si construieste feature-urile
- `train_test.py`: split train/test, label si normalizare
- `logistic_regression.py`: Logistic Regression manual
- `ml_signal.py`: transforma probabilitatile in semnal discret
- `q_learning.py`: agentul Q-learning
- `benchmark.py`: simulari si metrici pentru strategii
- `afisare.py`: afiseaza rezultatele in consola

## Instalare

```bash
pip install -r requirements.txt
```

Biblioteci folosite:
- `numpy`
- `pandas`

## Rulare

Pentru rezultatele principale:

```bash
python main.py
```
