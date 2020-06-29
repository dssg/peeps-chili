import os
import pickle
import datetime
from dateutil.relativedelta import relativedelta
import yaml
import pandas as pd
import sqlalchemy
import RecallAdjuster as ra


NUM_TRIALS = 5

bootstrap_weights = [
    {'W': 0.66, 'B': 0.27, 'H': 0.07}
    # {'W': 0.62, 'B': 0.27, 'H': 0.11},  # current
    # {'W': 0.58, 'B': 0.27, 'H': 0.15},
    # {'W': 0.54, 'B': 0.27, 'H': 0.19},
    # {'W': 0.50, 'B': 0.27, 'H': 0.23},
    # {'W': 0.46, 'B': 0.27, 'H': 0.27},
    # {'W': 0.42, 'B': 0.27, 'H': 0.31},
    # {'W': 0.38, 'B': 0.27, 'H': 0.35},
    # {'W': 0.34, 'B': 0.27, 'H': 0.39}
]

append_to = 'boostrap_hisp_frac_test.pkl'
export_file = 'boostrap_hisp_frac_test_new.pkl'


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

all_ts = []

for bs_wts in bootstrap_weights:
    for i in range(NUM_TRIALS):
        print('starting trial %s of %s with weights %s...' % (i, NUM_TRIALS, bs_wts))

        myRA = ra.RecallAdjuster(
                engine=conn,
                pg_role='kit',
                schema='kit_bias_class_test',
                experiment_hashes='09b3bcab5a6e1eb1c712571f6a5abb75',
                date_pairs=date_pairs,
                list_sizes=[500, 1000],
                #entity_demos='joco',
                entity_demos='kit_bias_class_test.entity_demos',
                demo_col='race_3way',
                bootstrap_weights=bs_wts
        )

        sql = """
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
               r.base_recall_b_to_h AS recall_b_to_h,
               r.base_frac_w AS frac_w,
               r.base_frac_b AS frac_b,
               r.base_frac_h AS frac_h
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
               r.adj_recall_b_to_h AS recall_b_to_h,
               r.adj_frac_w AS frac_w,
               r.adj_frac_b AS frac_b,
               r.adj_frac_h AS frac_h
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
               r.adj_recall_b_to_h AS recall_b_to_h,
               r.adj_frac_w AS frac_w,
               r.adj_frac_b AS frac_b,
               r.adj_frac_h AS frac_h
        FROM kit_bias_class_test.model_adjustment_results_race_3way r
        JOIN adj_mgs b
          ON r.model_group_id = b.model_group_id
          AND r.past_train_end_time = b.train_end_time
          AND r.list_size = b.list_size
          AND r.metric = b.metric
          AND r.parameter = b.parameter
        WHERE r.train_end_time > r.past_train_end_time

        UNION ALL

        -- Composite model (no decoupled models)
        SELECT 'Composite Model - Adjusted'::VARCHAR(128) AS strategy,
              train_end_time, past_train_end_time,
              list_size, metric, parameter,
              value,
              max_recall_ratio,
              recall_w_to_b,
              recall_w_to_h,
              recall_b_to_h,
              frac_w,
              frac_b,
              frac_h
        FROM kit_bias_class_test.composite_results_race_3way
        WHERE train_end_time > past_train_end_time
        ;
        """

        new_ts = pd.read_sql(sql, conn)
        new_ts['bootstrap_frac_w'] = bs_wts['W']
        new_ts['bootstrap_frac_b'] = bs_wts['B']
        new_ts['bootstrap_frac_h'] = bs_wts['H']
        new_ts['bootstrap_trial'] = i
        all_ts.append(new_ts)

if append_to:
    with open(append_to, 'rb') as f:
        old_res = pickle.load(f)
        all_ts.append(old_res)

results = pd.concat(all_ts)
with open(export_file, 'wb') as f:
    pickle.dump(results, f)

