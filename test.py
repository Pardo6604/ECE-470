from utils import load_environment_variables, get_api_key
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_environment_variables('.env')
key = get_api_key('ALPACA_API_KEY')
secret = get_api_key('ALPACA_API_SECRET')
client = StockHistoricalDataClient(key, secret)

portfolio = ["VZ", "UNH", "CVX", "CSCO", "IBM", 
    "MRK", "KO", "GS", "PG", "AMGN", 
    "HD", "JPM", "NKE", "JNJ", "MMM", 
    "MCD", "HON", "TRV", "DIS", "AXP", 
    "CAT", "MSFT", "V", "SHW", "CRM", 
    "WMT", "AAPL", "NVDA", "CASH", "USD", "OTHER"]

for sym in portfolio:
    try:
        request = StockBarsRequest(
            symbol_or_symbols=[sym],
            timeframe=TimeFrame.Day,
            start='2026-06-10',
            end='2026-06-11',
        )
        df = client.get_stock_bars(request).df
        if df.empty:
            print(f"{sym}: empty dataframe")
        elif 'timestamp' not in df.index.names:
            print(f"{sym}: no timestamp in index names. Index: {df.index}")
    except Exception as e:
        print(f"Error on {sym}: {e}")
