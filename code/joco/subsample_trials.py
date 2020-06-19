import os
import pickle
import datetime
from dateutil.relativedelta import relativedelta
import yaml
import pandas as pd
import sqlalchemy
import RecallAdjuster as ra

NUM_TRIALS = 10

base = datetime.datetime.strptime('2018-04-01', '%Y-%m-%d')
date_pairs = []
for x in range(9,-1,-1):
    date_pairs.append(
        (
        (base - relativedelta(months=4*x) - relativedelta(years=1)).strftime('%Y-%m-%d'),
        (base - relativedelta(months=4*x) - relativedelta(years=1)).strftime('%Y-%m-%d')
        )
    )
    date_pairs.append(
        (
        (base - relativedelta(months=4*x) - relativedelta(years=1)).strftime('%Y-%m-%d'),
        (base - relativedelta(months=4*x)).strftime('%Y-%m-%d')
        )
    )


def connect(poolclass=sqlalchemy.pool.QueuePool):
    with open(os.path.join(os.path.join('../..', 'config'), 'db_default_profile.yaml')) as fd:
        config = yaml.load(fd)
        dburl = sqlalchemy.engine.url.URL(
            "postgres",
            host=config["host"],
            username=config["user"],
            database=config["db"],
            password=config["pass"],
            port=config["port"],
        )
        return sqlalchemy.create_engine(dburl, poolclass=poolclass)

conn = connect()


all_fracs = []
all_ts = []

for i in range(NUM_TRIALS):
    print('starting trial %s of %s...' % (i, NUM_TRIALS))
    myRA = ra.RecallAdjuster(
            engine=conn,
            pg_role='johnson_county_ddj_write',
            schema='kit_bias_class_test',
            experiment_hashes='09b3bcab5a6e1eb1c712571f6a5abb75',
            date_pairs=date_pairs,
            list_sizes=[500],
            #entity_demos='joco',
            entity_demos='kit_bias_class_test.entity_demos',
            demo_col='race_3way',
            sample_weights={'W': 0.3, 'B': 0.6}
    )

    new_fracs = pd.read_sql("""
        SELECT train_end_time, COUNT(*) AS num_models,
               AVG(base_frac_b) AS avg_base_frac_b, AVG(base_frac_w) AS avg_base_frac_w, AVG(base_frac_h) AS avg_base_frac_h,
               AVG(adj_frac_b) AS avg_adj_frac_b, AVG(adj_frac_w) AS avg_adj_frac_w, AVG(adj_frac_h) AS avg_adj_frac_h
        FROM kit_bias_class_test.model_adjustment_results_race_3way
        WHERE base_value >= 0.45
        AND train_end_time > past_train_end_time
        GROUP BY 1
        ORDER BY 1 DESC;""", conn)
    new_fracs['trial'] = i
    all_fracs.append(new_fracs)

    ts_sql = """
        WITH mg_rns AS (
          SELECT *,
                 row_number() OVER (PARTITION BY train_end_time, list_size, metric, parameter ORDER BY base_value DESC, base_max_recall_ratio ASC, RANDOM()) AS rn_base,
                 row_number() OVER (PARTITION BY train_end_time, list_size, metric, parameter ORDER BY adj_value DESC, adj_max_recall_ratio ASC, RANDOM()) AS rn_adj
          FROM kit_bias_class_test.model_adjustment_results_race_3way
          WHERE past_train_end_time = train_end_time
        )
        , base_mgs AS (
          SELECT * FROM mg_rns WHERE rn_base = 1
        )
        , adj_mgs AS (
          SELECT * FROM mg_rns WHERE rn_adj = 1
        )
        -- Simple model selection on last time period, baseline with no recall adjustments
        SELECT 'Best Unadjusted Metric - Unadjusted'::VARCHAR(128) AS strategy,
               r.train_end_time, r.past_train_end_time,
               r.list_size, r.metric, r.parameter,
               r.base_value AS value,
               r.base_max_recall_ratio AS max_recall_ratio,
               r.base_recall_w_to_b AS recall_w_to_b,
               r.base_recall_w_to_h AS recall_w_to_h,
               r.base_recall_b_to_h AS recall_b_to_h
        FROM kit_bias_class_test.model_adjustment_results_race_3way r
        JOIN base_mgs b
          ON r.model_group_id = b.model_group_id
          AND r.past_train_end_time = b.train_end_time
          AND r.list_size = b.list_size
          AND r.metric = b.metric
          AND r.parameter = b.parameter
        WHERE r.train_end_time > r.past_train_end_time

        UNION ALL

        -- Model selection on last time before adjustment, with adjustment applied
        SELECT 'Best Unadjusted Metric - Adjusted'::VARCHAR(128) AS strategy,
               r.train_end_time, r.past_train_end_time,
               r.list_size, r.metric, r.parameter,
               r.adj_value AS value,
               r.adj_max_recall_ratio AS max_recall_ratio,
               r.adj_recall_w_to_b AS recall_w_to_b,
               r.adj_recall_w_to_h AS recall_w_to_h,
               r.adj_recall_b_to_h AS recall_b_to_h
        FROM kit_bias_class_test.model_adjustment_results_race_3way r
        JOIN base_mgs b
          ON r.model_group_id = b.model_group_id
          AND r.past_train_end_time = b.train_end_time
          AND r.list_size = b.list_size
          AND r.metric = b.metric
          AND r.parameter = b.parameter
        WHERE r.train_end_time > r.past_train_end_time

        UNION ALL

        -- Model selection on last time after adjustment, with adjustment applied
        SELECT 'Best Adjusted Metric - Adjusted'::VARCHAR(128) AS strategy,
               r.train_end_time, r.past_train_end_time,
               r.list_size, r.metric, r.parameter,
               r.adj_value AS value,
               r.adj_max_recall_ratio AS max_recall_ratio,
               r.adj_recall_w_to_b AS recall_w_to_b,
               r.adj_recall_w_to_h AS recall_w_to_h,
               r.adj_recall_b_to_h AS recall_b_to_h
        FROM kit_bias_class_test.model_adjustment_results_race_3way r
        JOIN adj_mgs b
          ON r.model_group_id = b.model_group_id
          AND r.past_train_end_time = b.train_end_time
          AND r.list_size = b.list_size
          AND r.metric = b.metric
          AND r.parameter = b.parameter
        WHERE r.train_end_time > r.past_train_end_time

        UNION ALL

        -- Composite model
        SELECT 'Composite Model - Adjusted'::VARCHAR(128) AS strategy,
              future_train_end_time AS train_end_time, past_train_end_time,
              list_size, metric, parameter,
              value,
              max_recall_ratio,
              recall_w_to_b,
              recall_w_to_h,
              recall_b_to_h
        FROM kit_bias_class_test.composite_results_race_3way
        WHERE future_train_end_time > past_train_end_time
        ;
        """
    new_ts = pd.read_sql(ts_sql, conn)
    all_ts.append(new_ts)

result_dfs = {
    'fracs': pd.concat(all_fracs),
    'ts': pd.concat(all_ts)
}

with open('multi_sample.pkl', 'wb') as f:
    pickle.dump(result_dfs, f)


