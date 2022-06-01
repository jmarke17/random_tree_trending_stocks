import yfinance as yf
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from datetime import datetime

#Collecting data from Yahoo Finance. For future iterations, we wil keep the data in csv. 
#As the data changes month to month, we will add the month to file name.

stock_name = "O"
file_name = stock_name + "_{}".format(datetime.now().year)+".csv"

if os.path.exists(stock_name):
    stock = pd.read_csv(file_name, index_col=0)
else:
    stock = yf.Ticker(stock_name)
    stock = stock.history(period="max")
    stock.to_csv(file_name)
    
    
# We will prepare the data

del stock["Dividends"],stock["Stock Splits"]
stock.index = pd.to_datetime(stock.index)
stock["Price Tomorrow"] = stock["Close"].shift(-1)
stock["Target"] = (stock["Price Tomorrow"] > stock["Close"]).astype(int)
stock = stock.loc["1990-01-01":].copy() 


#Iniciamos el modelo

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >=.6] = 1
    preds[preds <.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=2500, step=100):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)

horizons = [2,5,60,250,1000]
predictors = []

for horizon in horizons:
    rolling_averages = stock.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    stock[ratio_column] = stock["Close"] / rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    stock[trend_column] = stock.shift(1).rolling(horizon).sum()["Target"]
    
    predictors+= [ratio_column, trend_column]

stock = stock.dropna()

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
predictions = backtest(stock, model, predictors)

print("Using a Random Tree Classifier on the stock "+stock_name+" ")
print("The predictors that we have used are the price above the different means => "+str(horizons))
print("We have reach a precision score of " +str((round(precision_score(predictions["Target"], predictions["Predictions"]),3)))+" %")

