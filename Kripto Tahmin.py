
get_ipython().system('pip install yfinance pandas numpy scikit-learn keras tensorflow')


# In[13]:


import requests
import pandas as pd
from datetime import datetime, timedelta

def get_binance_data(symbol, interval, start_date, end_date):
    # Convert dates to milliseconds since Unix epoch
    start_time = int(start_date.timestamp() * 1000)
    end_time = int(end_date.timestamp() * 1000)
    
    # Construct the URL for Binance API
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={start_time}&endTime={end_time}'
    
    # Make request to Binance API
    response = requests.get(url)
    data = response.json()
    
    # Create DataFrame from API response
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                     'close_time', 'quote_asset_volume', 'number_of_trades', 
                                     'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    
    # Convert timestamp to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    # Select relevant columns and convert to float
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    
    return df

# Örneğin, son 6 ayın verilerini çekelim
end_date = datetime.now()
start_date = end_date - timedelta(days=180)  # 180 gün önce
symbol = 'BTCUSDT'
interval = '1d'  # Günlük veri

# Veri çekme
data = get_binance_data(symbol, interval, start_date, end_date)
print(data.head())



# In[14]:


import numpy as np

def create_features(df):
    df['return'] = df['close'].pct_change()
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['volatility'] = df['close'].rolling(window=20).std()
    df.dropna(inplace=True)
    return df

# Özellik mühendisliği işlemleri
data = create_features(data)
print(data.head())



# In[19]:


import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input

def prepare_data(df, prediction_days):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['close']].values)
    
    X, y = [], []
    for i in range(prediction_days, len(scaled_data)):
        X.append(scaled_data[i-prediction_days:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Model için hazırlık
prediction_days = 30  # 30 gün öncesinden tahmin yapılacak
X, y, scaler = prepare_data(data, prediction_days)

# Model oluşturma
model = Sequential()
model.add(Input(shape=(X.shape[1], X.shape[2])))  # Girdi şekli belirleme
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitme
model.fit(X, y, epochs=50, batch_size=32)

# Tahminler yapma
future_days = 60  # 60 gün sonrası için tahmin yapılacak
predicted_prices = []

last_sequence = X[-1]  # Son gözlem dizisi
for _ in range(future_days):
    prediction = model.predict(np.array([last_sequence]))[0, 0]
    predicted_prices.append(prediction)
    last_sequence = np.roll(last_sequence, -1)
    last_sequence[-1] = prediction

predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Sonuçların görselleştirilmesi
import matplotlib.pyplot as plt

# Gerçek verilerin grafiği
plt.figure(figsize=(14, 7))
plt.plot(data.index[-len(predicted_prices)-1:], data['close'].values[-len(predicted_prices)-1:], label='Gerçek Fiyatlar', color='blue')

# Tahmin edilen fiyatların grafiği
future_dates = pd.date_range(start=data.index[-1], periods=future_days+1)
plt.plot(future_dates[:-1], predicted_prices, label='Tahmini Fiyatlar', color='red')  # future_dates dizisinden son elemanı çıkar

plt.title(f'{symbol} Fiyat Tahmini')
plt.xlabel('Zaman')
plt.ylabel('Fiyat (USDT)')
plt.legend()
plt.show()






