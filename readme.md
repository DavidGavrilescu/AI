# proiect ai

proiectul lucreaza cu date pentru SPY si incearca sa compare cateva strategii simple de trading.

ideea principala este:
- iau datele din yfinance
- fac cateva feature-uri din pret si volum
- antrenez un Logistic Regression
- transform rezultatul lui intr-un semnal simplu
- folosesc semnalul asta si in Q-learning
- la final compar strategiile pe datele de test

strategii comparate:
- buy and hold
- LR only
- random agent
- ML + Q-learning
- Q-learning fara ML

date:
- ticker: SPY
- perioada: 2010-01-01 -> 2019-01-01
- sursa datelor: yfinance
- fisier local: `spy_data.csv`

datele se salveaza in `spy_data.csv`, ca sa nu fie descarcate de fiecare data.

pipeline:
- incarca datele SPY
- calculeaza feature-urile
- imparte datele in train si test
- face label-ul pentru Logistic Regression
- normalizeaza feature-urile
- antreneaza Logistic Regression
- transforma probabilitatea in `ml_signal`
- antreneaza Q-learning pe train
- testeaza strategiile pe test
- afiseaza benchmark-ul

feature-uri:
- `daily_return`: randamentul fata de ziua precedenta
- `open_close_return`: cat s-a miscat pretul de la Open la Close
- `high_low_range`: diferenta High-Low raportata la Close
- `volume_change`: schimbarea volumului
- `ma_ratio`: diferenta dintre media mobila pe 5 zile si cea pe 20 de zile

mai sunt calculate si:
- `trend`: 0 = descendent, 1 = neutru, 2 = ascendent
- `future_return_5d`: randamentul peste 5 zile

`future_return_5d` nu este folosit ca feature pentru model. Este folosit doar pentru label si pentru recompensa din Q-learning.

train/test:
- primele 70% din date sunt train
- ultimele 30% din date sunt test
- `ml_label` se face dupa mediana lui `future_return_5d` din train
- normalizarea se face cu min si max din train

Logistic Regression:
- este facut manual cu numpy
- foloseste sigmoid
- ponderile incep de la 0
- se antreneaza cu gradient descent
- prezice daca randamentul pe 5 zile este peste mediana din train
- pragul de predictie este 0.5

`ml_signal`:
- 0 daca probabilitatea e sub prag
- 1 daca probabilitatea e in zona de incertitudine
- 2 daca probabilitatea e peste prag

zona de incertitudine se seteaza din `SIGNAL_MARGIN`.

Q-learning:
- starea este formata din `ml_signal`, pozitie si `trend`
- pozitia este cash sau SPY
- actiunile sunt BUY, SELL, HOLD
- daca agentul e cash, poate cumpara sau face HOLD
- daca agentul are SPY, poate vinde sau face HOLD
- recompensa vine din `future_return_5d`
- tranzactiile au cost
- cash-ul este penalizat putin cand piata urca
- epsilon scade treptat, deci agentul exploreaza mai mult la inceput

am pus doua variante:
- `ML + Q-learning`: foloseste `ml_signal` + pozitie + trend
- `Q-learning fara ML`: ignora semnalul ML si foloseste doar pozitia + trendul

benchmark:
- toate strategiile se ruleaza pe test
- decizia de azi se executa maine
- portofoliul incepe cu 100000 cash
- costul de tranzactie este 0.05%
- random agent se ruleaza de mai multe ori si se ia media

in benchmark se afiseaza:
- valoare finala
- randament %
- tranzactii
- expunere %
- Sharpe Ratio
- Max Drawdown %

fisiere:
- `main.py`: ruleaza tot pipeline-ul
- `config.py`: tine parametrii proiectului
- `data_processing.py`: incarca datele si construieste feature-urile
- `train_test.py`: imparte datele, face label-ul si normalizeaza
- `logistic_regression.py`: antreneaza si evalueaza Logistic Regression
- `ml_signal.py`: transforma probabilitatea LR in semnal discret
- `q_learning.py`: partea de Q-learning
- `benchmark.py`: simuleaza strategiile si calculeaza metricile
- `afisare.py`: afiseaza rezultatele modelului

instalare:
```bash
pip install -r requirements.txt
```

ruleaza:
```bash
python main.py
```

output-ul afiseaza:
- intervalele train/test
- feature-urile LR
- baseline train/test
- accuracy train/test
- bias-ul si ponderile LR
- cum se face `ml_signal`
- tabelul de benchmark
