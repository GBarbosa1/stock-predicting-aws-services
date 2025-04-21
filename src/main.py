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
    """
    Run an Athena query and return the results as a pandas DataFrame.
    
    Parameters
    ----------
    query : str
        The SQL query to execute.
    database : str
        The Athena database to run the query against.
    output_s3_path : str
        S3 location where Athena will dump the query results, e.g.
        "s3://your-bucket/athena-results/".
    region : str, default "us-east-1"
        AWS region for Athena.
    poll_interval : float, default 2.0
        Seconds to wait between status checks.
    
    Returns
    -------
    pd.DataFrame
        The full result set, with the first row used as column headers.
    """
    athena = boto3.client("athena", region_name=region)
    # 1) start the query
    resp = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output_s3_path}
    )
    qid = resp["QueryExecutionId"]

    # 2) poll until the query finishes
    while True:
        status = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]["State"]
        if status in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(poll_interval)

    if status != "SUCCEEDED":
        raise RuntimeError(f"Athena query {qid} did not succeed: {status}")

    # 3) fetch all rows (including header)
    paginator = athena.get_paginator("get_query_results")
    rows = []
    for page in paginator.paginate(QueryExecutionId=qid):
        for r in page["ResultSet"]["Rows"]:
            # Each 'r' has a list of {'VarCharValue': value}
            rows.append([c.get("VarCharValue") for c in r["Data"]])

    # 4) build DataFrame: first row = columns, rest = data
    header, data = rows[0], rows[1:]
    df = pd.DataFrame(data, columns=header)
    return df

# Example usage
if __name__ == "__main__":
    df = run_athena_query_df(
        query="SELECT * FROM my_table LIMIT 10",
        database="s3silver_finance_data",
        output_s3_path="s3://silver-finance-data/athena_querie_results/",
        region="us-east-1"
    )
    print(df.head())
