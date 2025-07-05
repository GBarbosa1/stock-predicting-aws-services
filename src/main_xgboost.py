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

tickers_table = os.environ['tickers_table']
finance_database = os.environ['finance_database']
athena_query_result = os.environ['athena_query_result']
region  = os.environ['region']
instance_id = os.environ['inscance_id']

logging.basicConfig(level=logging.ERROR)

def put_files_to_s3(bucketname:str, json_data):
    s3 = boto3.client('s3')
    s3.put_object(
    Bucket = bucketname,
    Key = str(uuid.uuid4()),
    Body=json_data,
    ContentType="application/json")

def terminate_self(instance_id):
    instance_id = instance_id()
    ec2 = boto3.client('ec2', region_name='us-east-1')
    ec2.terminate_instances(InstanceIds=[instance_id])
    print(f"Termination initiated for instance {instance_id}")

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
    dtest = xgb.DMatrix(df)
    loaded_model = XGBClassifier()
    loaded_model = pickle.load(open(model, "rb"))
    predictions = loaded_model.predict(dtest)
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
    df['date_capture'] = pd.to_datetime(df['date_capture'])
    df = df.sort_values(by='date_capture', ascending=True)
    df['SMA_10'] = trend.sma_indicator(df['Close'], window=10)
    df['SMA_50'] = trend.sma_indicator(df['Close'], window=50)
    df['EMA_10'] = trend.ema_indicator(df['Close'], window=10)
    df['EMA_50'] = trend.ema_indicator(df['Close'], window=50)
    df['MACD'] = trend.macd(df['Close'])
    df['MACD_Signal'] = trend.macd_signal(df['Close'])
    df['RSI'] = momentum.rsi(df['Close'], window=14)
    df['Stochastic_K'] = momentum.stoch(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['Stochastic_D'] = momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['ATR'] = volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['Bollinger_High'] = volatility.bollinger_hband(df['Close'], window=20, window_dev=2)
    df['Bollinger_Low'] = volatility.bollinger_lband(df['Close'], window=20, window_dev=2)
    

    df.dropna(inplace=True)
    lag_days = 30

    for lag in range(1, lag_days + 1):
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)

    for lag in range(1, lag_days + 1):
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)

    df.dropna(inplace=True)

    df['target'] = df['Close'].shift(-1)

    feature_columns = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50',
    'MACD', 'MACD_Signal', 'RSI',
    'Stochastic_K', 'Stochastic_D',
    'ATR', 'Bollinger_High', 'Bollinger_Low']

    for lag in range(1, lag_days + 1):
        feature_columns.append(f'Close_Lag_{lag}')
        feature_columns.append(f'Volume_Lag_{lag}')

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
        region='us-east-1'
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
            region="us-east-1"
        )
        numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce').astype(float)
        data, feature_columns = create_features(data)
        data.drop(columns=['date_capture','target'], inplace = True)
        data.apply(pd.to_numeric, errors='coerce').astype(float)
        predictions = make_predictions(df = data[feature_columns].tail(10), model = f'src/xgboost/xgb_fin_model_v1_{ticker}.pkl')
    
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
        terminate_self(instance_id)