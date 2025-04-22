import boto3
import pandas as pd
import time

def run_athena_query_df(
    query: str,
    database: str,
    output_s3_path: str,
    region: str = "us-east-1",
    poll_interval: float = 2.0
) -> pd.DataFrame:
    
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
        print(rows)
