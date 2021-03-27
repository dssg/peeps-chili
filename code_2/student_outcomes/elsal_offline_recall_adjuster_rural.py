import os
import yaml
import pandas as pd
import sqlalchemy
import datetime
from dateutil.relativedelta import relativedelta
import time
import sys
import RecallAdjuster as ra

def connect(poolclass=sqlalchemy.pool.QueuePool):
    with open(os.path.join('..','config', 'elsal_db_profile2.yaml')) as fd:
        config = yaml.load(fd)
        dburl = sqlalchemy.engine.url.URL(
            "postgres",
            host = config["host"],
            username = config["user"],
            database = config["db"],
            password = config["pass"],
            port = config["port"]
        )
        return sqlalchemy.create_engine(dburl, poolclass=poolclass)

def get_date_pairs(base_date, n_intervals):
    date_pairs = []
    base = datetime.datetime.strptime(base_date, '%Y-%m-%d')

    for x in range(n_intervals, -1, -1):
        date_pairs.append(
            (
                (base - relativedelta(months=12*x) - relativedelta(months=12)).strftime('%Y-%m-%d'),
                (base - relativedelta(months=12*x) - relativedelta(months=12)).strftime('%Y-%m-%d')
            )
        )
        date_pairs.append(
            (
                (base - relativedelta(months=12*x) - relativedelta(months=12)).strftime('%Y-%m-%d'),
                (base - relativedelta(months=12*x)).strftime('%Y-%m-%d')
            )
        )

    return date_pairs

def get_model_groups(conn):
    query = """
    select distinct(model_id), model_group_id
    from test_results.predictions 
    join model_metadata.models using (model_id)
    where model_group_id NOT IN (82,83,84,85,76,77,78,79,80,96,101,102,104,105,106,107,108,109,110,116,122,123,124,125,126,127,128,133,134,136)
    """
    df = pd.read_sql(query, conn)
    model_groups = df['model_group_id'].values

    return model_groups

def run_RA(date_pairs, model_groups):
    start_time = time.time()
    print("Running Recall Adjuster with ")
    print("[ListSizes]:"+str([30000]))
    print("[ModelGroups]:"+str(model_groups))
    
    myRA = ra.RecallAdjuster(
        engine=conn,
        pg_role='postgres',
        schema='bias_analysis_1year_all_models_rural',
        experiment_hashes='None',
        date_pairs=date_pairs,
        list_sizes=[10000,30000],
        entity_demos='bias_analysis_1year.entity_demos_two',
        demo_col='rural',
        model_group_ids = model_groups,
        dataset = 'elsal'
    )
    print("Time Taken="+str(time.time() - start_time))

if __name__ == "__main__":
    conn = connect()
    base_date = '2016-01-01'
    date_pairs = get_date_pairs(base_date, 6)
    model_groups = get_model_groups(conn)
    run_RA(date_pairs, model_groups)
