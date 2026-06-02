# proiect ai

proiectul ia datele pentru SPY si face niste feature-uri simple din pret si volum.

dupa aia:
- imparte datele in train si test
- antreneaza Logistic Regression
- transforma predictia LR in `ml_signal`
- foloseste `ml_signal` + trendul in Q-learning
- compara strategiile pe test

strategii comparate:
- buy and hold
- LR only
- random agent
- ML + Q-learning

ruleaza:
python main.py
