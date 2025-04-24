import boto3
import pandas as pd
import time
from ta import trend, momentum, volatility
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

def make_predictions(df, model):
    loaded_model = XGBClassifier()
    loaded_model.load_model(model)
    predictions = loaded_model.predict(df)
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
    df['Bollinger_high'] = volatility.bollinger_hband(df['Close'], window=20, window_dev=2)
    df['Bollinger_Low'] = volatility.bollinger_lband(df['Close'], window=20, window_dev=2)
    

    df.dropna(inplace=True)
    lag_days = 30

    for lag in range(1, lag_days + 1):
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)

    for lag in range(1, lag_days + 1):
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)

    df['target'] = df['Close'].shift(-1)
    return df

if __name__ == "__main__":
    feature_columns = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50',
    'MACD', 'MACD_Signal', 'RSI',
    'Stochastic_K', 'Stochastic_D',
    'ATR', 'Bollinger_High', 'Bollinger_Low']
    
    tickers_to_query = run_athena_query_df(
        query="""
        select distinct
        partition_0
        from finance.s3silver_finance_data
        where cast(date_capture as date) >= date_add('day',-5,cast(date_capture as date));
        """,
        database="s3silver_finance_data",
        output_s3_path="s3://silver-finance-data/athena_querie_results/",
        region="us-east-1"
    )
    for index, rows in tickers_to_query.iterrows():
        ticker = rows['partition_0']
        data = run_athena_query_df(
            query=f"""
            select
            cast(date_capture as date) as date_capture,
            cast(close as double) as Close,
            cast(high as double) as High,
            cast(low as double) as Low,
            cast(open as double) as Open,
            cast(volume as int) as Volume,
            partition_0
            from finance.s3silver_finance_data 
            where 1 = 1
            and partition_0 = '{ticker}'
            and cast(date_capture as date) >= date_add('day',-360,current_date)
            order by cast(date_capture as date) desc
            limit 120;
            """,
            database="s3silver_finance_data",
            output_s3_path="s3://silver-finance-data/athena_querie_results/",
            region="us-east-1"
        )
        data[columns_to_cast] = data[feature_columns].astype(float)
        data = create_features(data)
        print(data.columns)
        make_predictions(data[feature_columns].tail(10), 'src/xgboost/finance_xgboost.json')
        print(make_predictions)
