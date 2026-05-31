import yfinance as yf

TICKER = "AAPL"

data = yf.download(
    TICKER,
    start="2010-01-01",
    end="2020-12-31",
    auto_adjust=True,
    progress=False, 
    multi_level_index=False # fara sa zica AAPL
)

data = data.reset_index() # sa apara coloanele din header fara spatiu gol sub

print(data.head(3)) # doar primele 3 rnduri
