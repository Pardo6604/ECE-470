from utils import load_environment_variables, get_api_key
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
import ta


load_environment_variables('.env')
key = get_api_key('ALPACA_API_KEY')
secret = get_api_key('ALPACA_API_SECRET')

portfolio = [
    'VZ', 'UNH', 'CVX', 'CSCO', 'IBM', 
    'MRK', 'KO', 'GS', 'PG', 'AMGN', 
    'HD', 'JPM', 'NKE', 'JNJ', 'MMM', 
    'MCD', 'HON', 'TRV', 'DIS', 'AXP', 
    'CAT', 'MSFT', 'V', 'SHW', 'CRM',
    'WMT', 'AAPL', 'NVDA'] # Missing USD and AGPXX since no data is found

client = StockHistoricalDataClient(key, secret)

def get_data(symbol, timeframe, start_date, end_date):
    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=timeframe,
        start=start_date,
        end=end_date,
    )
    data = client.get_stock_bars(request).df
    
    if data.empty:
        return pd.DataFrame(columns=['Symbol', 'Open', 'High', 'Low', 'Close', 'Volume'])

    data.index = data.index.get_level_values('timestamp').tz_convert('America/New_York')
    data = data[['open', 'high', 'low', 'close', 'volume']]
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    data['Symbol'] = symbol
    data = data.reindex(columns=['Symbol', 'Open', 'High', 'Low', 'Close', 'Volume'])
    return data

def get_tech_indicators(data):
    if data.empty:
        return data
    data = ta.add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    return data

def portfolio_data(portfolio, timeframe, start, end):
    data = {}
    for symbol in portfolio:
        OHLCV_data = get_data(symbol, timeframe, start, end)
        data[symbol] = get_tech_indicators(OHLCV_data)
    

    return data

data = portfolio_data(portfolio, TimeFrame.Day, "2026-01-01", "2026-06-11")
print(data['VZ'])





    
