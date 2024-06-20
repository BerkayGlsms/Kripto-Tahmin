!pip install yfinance pandas numpy scikit-learn keras tensorflow

from flask import Flask, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

app = Flask(__name__)

# Veri çekme ve özellikleri oluşturma fonksiyonları
def get_binance_data(symbol, interval, limit=1000):
    import requests
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                     'close_time', 'quote_asset_volume', 'number_of_trades',
                                     'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def create_features(df):
    df['return'] = df['close'].pct_change()
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['volatility'] = df['close'].rolling(window=20).std()
    df.dropna(inplace=True)
    return df

def prepare_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['close']].values)

    X = []
    y = []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

# Model oluşturma
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitme (örnek olarak burada eğitim verisi çekiliyor, gerçek zamanlı uygulamada farklı yöntemler gerekebilir)
symbol = 'BTCUSDT'
interval = '1h'
data = get_binance_data(symbol, interval)
data = create_features(data)
X, y, scaler = prepare_data(data)
model.fit(X, y, epochs=50, batch_size=32)

# Flask API Endpoint
@app.route('/predict', methods=['GET'])
def predict():
    # Önümüzdeki 60 gün için tahminler yap
    future_days = 60
    predicted_prices = []

    last_sequence = X[-1]  # Son gözlem dizisi
    for _ in range(future_days):
        prediction = model.predict(np.array([last_sequence]))[0, 0]
        predicted_prices.append(prediction)
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = prediction

    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

    # Tahminleri JSON formatında dön
    future_dates = pd.date_range(start=data.index[-1], periods=future_days+1)
    results = {
        'dates': future_dates.strftime('%Y-%m-%d').tolist(),
        'predicted_prices': predicted_prices.flatten().tolist()
    }

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)



