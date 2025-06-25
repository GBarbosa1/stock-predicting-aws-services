import boto3
import pandas as pd
import time
import xgboost as xgb
import numpy as np
import pickle
import logging
import json
import uuid
import os
from ta import trend, momentum, volatility
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import numpy as np


tickers_table = os.environ['tickers_table']
finance_database = os.environ['finance_database']
athena_query_result = os.environ['athena_query_result']
region  = os.environ['region']

logging.basicConfig(level=logging.ERROR)

def put_files_to_s3(bucketname:str, json_data):
    s3 = boto3.client('s3')
    s3.put_object(
    Bucket = bucketname,
    Key = str(uuid.uuid4()),
    Body=json_data,
    ContentType="application/json")


def get_next_weekdays(start_date, num_days):
    days = []
    current_day = start_date
    while len(days) < num_days:
        if current_day.weekday() < 5:  # Monday to Friday
            days.append(current_day)
        current_day += timedelta(days=1)
    return days

def make_predictions(df, model):
    logging.info(f"Using the following model: {model}")
    model = load_model(model)
    return predictions

def run_athena_query_df(
    query: str,
    database: str,
    output_s3_path: str,
    region: str = "us-east-1",
    poll_interval: float = 2.0) -> pd.DataFrame:
    
    athena = boto3.client("athena", region_name=region)
    resp = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output_s3_path}
    )
    qid = resp["QueryExecutionId"]

    while True:
        status = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]["State"]
        if status in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(poll_interval)

    if status != "SUCCEEDED":
        raise RuntimeError(f"Athena query {qid} did not succeed: {status}")

    paginator = athena.get_paginator("get_query_results")
    rows = []
    for page in paginator.paginate(QueryExecutionId=qid):
        for r in page["ResultSet"]["Rows"]:
            rows.append([c.get("VarCharValue") for c in r["Data"]])

    header, data = rows[0], rows[1:]
    df = pd.DataFrame(data, columns=header)
    return df

def create_features(df):
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['RSI'] = momentum.rsi(df['Close'], window=14)
    df['Stochastic_K'] = ((df['Close'] - df['Low'].rolling(window=14).min()) / (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min())) * 100
    df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
    df['ATR'] = df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min()
    df['Bollinger_High'] = df['SMA_10'] + 2 * df['Close'].rolling(window=10).std()
    df['Bollinger_Low'] = df['SMA_10'] - 2 * df['Close'].rolling(window=10).std()

    df.dropna(inplace=True)

    feature_columns = [
      'Open', 'High', 'Low', 'Close', 'Volume',
      'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50',
      'MACD', 'MACD_Signal', 'RSI',
      'Stochastic_K', 'Stochastic_D',
      'ATR', 'Bollinger_High', 'Bollinger_Low']
    
    features = df[feature_columns].values
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    return df, feature_columns

if __name__ == "__main__":
    feature_columns = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50',
    'MACD', 'MACD_Signal', 'RSI',
    'Stochastic_K', 'Stochastic_D',
    'ATR', 'Bollinger_High', 'Bollinger_Low']
    
    tickers_to_query = run_athena_query_df(
        query=f"""
        select distinct
        partition_0
        from {tickers_table}
        where cast(date_capture as date) >= date_add('day',-5,cast(date_capture as date));
        """,
        database=f"{finance_database}",
        output_s3_path=f"{athena_query_result}",
        region=f"{us-east-1}"
    )
    for index, rows in tickers_to_query.iterrows():
        ticker = rows['partition_0']
        data = run_athena_query_df(
            query=f"""
            select
            cast(date_capture as date) as date_capture,
            cast(open as double) as Open,
            cast(high as double) as High,
            cast(low as double) as Low,
            cast(close as double) as Close,
            cast(volume as int) as Volume
            from {tickers_table}
            where 1 = 1
            and partition_0 = '{ticker}'
            and cast(date_capture as date) >= date_add('day',-360,current_date)
            order by cast(date_capture as date) desc
            limit 120;
            """,
            database=f"{finance_database}",
            output_s3_path=f"{athena_query_result}",
            region=f"{us-east-1}"
        )
        numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce').astype(float)
        data, feature_columns = create_features(data)
        data.drop(columns=['date_capture','target'], inplace = True)
        data.apply(pd.to_numeric, errors='coerce').astype(float)
        predictions = make_predictions(df = data[feature_columns].tail(10), model = f'src/lstm/xgb_fin_model_v1_{ticker}.h5')
    
        today = datetime.now().date()
        today_string = today.strftime("%Y-%m-%d")

        if today.weekday() == 5:
            today += timedelta(days=2)
        elif today.weekday() == 6:
            today += timedelta(days=1)

        prediction_dates = get_next_weekdays(today, len(predictions))

        df_predictions = pd.DataFrame({
            'DATE': prediction_dates,
            'PRICE_PREDICTION': predictions,
            'CAPTURE': today_string,
            'TICKER': ticker
        })
        json_list = df_predictions.apply(lambda row: {
            'date': row['DATE'].strftime("%Y-%m-%d"),
            'price_prediction': row['PRICE_PREDICTION'],
            'capture': row['CAPTURE'],
            'ticker':row['TICKER']
        }, axis=1).tolist()
        for index, json_obj in enumerate(json_list):
            json_str = json.dumps(json_obj)
            capture_date = json_obj['capture']
            predict_date = json_obj['date']
            put_files_to_s3('gold-finance-data',json_str)