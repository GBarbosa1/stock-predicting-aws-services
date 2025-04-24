import boto3
import pandas as pd
import time
from ta import trend, momentum, volatility

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
    df['SMA_10'] = trend.sma_indicator(df['close'], window=10)
    df['SMA_50'] = trend.sma_indicator(df['close'], window=50)
    df['EMA_10'] = trend.ema_indicator(df['close'], window=10)
    df['EMA_50'] = trend.ema_indicator(df['close'], window=50)
    df['MACD'] = trend.macd(df['close'])
    df['MACD_Signal'] = trend.macd_signal(df['close'])
    df['RSI'] = momentum.rsi(df['close'], window=14)
    df['Stochastic_K'] = momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['Stochastic_D'] = momentum.stoch_signal(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['ATR'] = volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['Bollinger_high'] = volatility.bollinger_hband(df['close'], window=20, window_dev=2)
    df['Bollinger_low'] = volatility.bollinger_lband(df['close'], window=20, window_dev=2)
    df.dropna(inplace=True)
    lag_days = 30

    for lag in range(1, lag_days + 1):
        df[f'close_Lag_{lag}'] = df['close'].shift(lag)

    for lag in range(1, lag_days + 1):
        df[f'Volume_Lag_{lag}'] = df['volume'].shift(lag)

    df['target'] = df['close'].shift(-1)
    return df

if __name__ == "__main__":
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
            cast(close as double) as close,
            cast(high as double) as high,
            cast(low as double) as low,
            cast(open as double) as open,
            cast(volume as int) as volume,
            partition_0
            from finance.s3silver_finance_data 
            where 1 = 1
            and partition_0 = '{ticker}'
            and cast(date_capture as date) >= date_add('day',-360,current_date)
            order by cast(date_capture as date) asc
            limit 120;
            """,
            database="s3silver_finance_data",
            output_s3_path="s3://silver-finance-data/athena_querie_results/",
            region="us-east-1"
        )
        columns_to_cast = ['close', 'high', 'low','open']
        data[columns_to_cast] = data[columns_to_cast].astype(float)
        data = create_features(data)
        print(data.head(10))
        print(data.tail(10))


