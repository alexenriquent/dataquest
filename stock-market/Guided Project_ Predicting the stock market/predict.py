import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("sphist.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values('Date', ascending=True)

df['day_5'] = df['Close'].rolling(5).mean()
df['day_5'] = df['day_5'].shift(1)

df['day_30'] = df['Close'].rolling(30).mean()
df['day_30'] = df['day_30'].shift(1)

df['day_365'] = df['Close'].rolling(365).mean()
df['day_365'] = df['day_365'].shift(1)

df = df[df["Date"] > datetime(year=1951, month=1, day=3)]
df = df.dropna(axis=0)

train = df[df["Date"] < datetime(year=2013, month=1, day=1)]
test = df[df["Date"] > datetime(year=2013, month=1, day=1)]

def train_and_test(col) :
    lm = LinearRegression()
    lm.fit(train[[col]], train['Close'])
    predictions = lm.predict(test[[col]])
    mse = mean_squared_error(test['Close'], predictions)
    return mse

print("5:", train_and_test('day_5'))
print("30:", train_and_test('day_30'))
print("365:", train_and_test('day_365'))
