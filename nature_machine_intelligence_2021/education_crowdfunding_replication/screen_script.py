import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

import os
import yaml
import sqlalchemy


from itertools import permutations
from jinja2 import Template
import dateparser


from ohio.ext.numpy import pg_copy_to_table

date_pairs_all = [
 ('2011-03-01', '2011-03-01'),
 ('2011-03-01', '2011-07-01'),
 ('2011-05-01', '2011-05-01'),
 ('2011-05-01', '2011-09-01'),
 ('2011-07-01', '2011-07-01'),
 ('2011-07-01', '2011-11-01'),
 ('2011-09-01', '2011-09-01'),
 ('2011-09-01', '2012-01-01'),
 ('2011-11-01', '2011-11-01'),
 ('2011-11-01', '2012-03-01'),
 ('2012-01-01', '2012-01-01'),
 ('2012-01-01', '2012-05-01'),
 ('2012-03-01', '2012-03-01'),
 ('2012-03-01', '2012-07-01'),
 ('2012-05-01', '2012-05-01'),
 ('2012-05-01', '2012-09-01'),
 ('2012-07-01', '2012-07-01'),
 ('2012-07-01', '2012-11-01'),
 ('2012-09-01', '2012-09-01'),
 ('2012-09-01', '2013-01-01')
 ]
date_list = ['2011-03-01', '2011-05-01', '2011-07-01', '2011-09-01', '2011-11-01', '2012-01-01', '2012-03-01', '2012-05-01', '2012-07-01', '2012-09-01', '2012-11-01', '2013-01-01']
date_weights = {}
for i, date in enumerate(date_list):
    if i <= 1:
        date_weights[date] = {date_list[max(0, i-1)]: 1.0, "past_train_end_time": date_list[max(0, i-1)]}
    else:
        date_weights[date] = {date_list[i-1]: 0.99, date_list[i-2]: 0.01, "past_train_end_time": date_list[i-1]}



with open('db_profile.yaml') as fd:
    config = yaml.full_load(fd)
    dburl = sqlalchemy.engine.url.URL(
        "postgresql",
        host=config["host"],
        username=config["user"],
        database=config["db"],
        password=config["pass"],
        port=config["port"],
    )
    engine_donors = sqlalchemy.create_engine(dburl, poolclass=sqlalchemy.pool.QueuePool)
    
    
def validate_dates(params):
    for past, future in params['date_pairs']:
        if dateparser.parse(past) > dateparser.parse(future):
            raise ValueError('Error! Cannot validate on the past. %s should be no earlier than %s.' % (future, past))

                
ENTITY_DEMO_FILES = {
    'joco': {
        'sql_tmpl': 'joco_entity_demos.sql.tmpl',
        'check_sql': """
            WITH all_matches AS (
                SELECT COUNT(DISTINCT ((mg.model_config->'matchdatetime')::VARCHAR)::TIMESTAMP) AS num_match
                FROM tmp_bias_models
                JOIN model_metadata.model_groups mg USING(model_group_id)
            )
            SELECT num_match = 1 AS pass_check
            FROM all_matches
        """
        }
}
def create_entity_demos(engine, params, entity_demos):
    sql_file = ENTITY_DEMO_FILES[entity_demos]['sql_tmpl']
    sql = Template(open(sql_file, 'r').read()).render(params)
    self.engine.execute(sql)
    self.engine.execute("COMMIT")

    # consistency check:
    check_sql = ENTITY_DEMO_FILES[entity_demos]['check_sql']
    if not engine.execute(check_sql).fetchall()[0][0]:
        raise RuntimeError('Entity Demos failed consistency check:\n %s' % check_sql)

    return '%s.entity_demos' % params['schema']

engine_donors.execute('TRUNCATE TABLE bias_results.composite_results_plevel;')
engine_donors.execute('TRUNCATE TABLE bias_results.model_adjustment_results_plevel;')
engine_donors.execute('COMMIT;')


def get_weight_case_str(date_weights):
    if len(date_weights) == 0:
        return "0"
    s = "CASE"
    for future_train_end_time in date_weights:
        for train_end_time in date_weights[future_train_end_time]:
            if train_end_time == "past_train_end_time":
                pass
            else:
                w = date_weights[future_train_end_time][train_end_time]
                s += f" WHEN future_train_end_time = '{future_train_end_time}' AND train_end_time = '{train_end_time}' THEN {w} "
    s += "ELSE 0 END"
    return s

def get_weight_past_train_end_time_case_str(date_weights):
    if len(date_weights) == 0:
        return f"'date_list[0]'::TIMESTAMP"
    s = "CASE"
    for future_train_end_time in date_weights:
        past_train_end_time = date_weights[future_train_end_time]["past_train_end_time"]
        s += f" WHEN future_train_end_time = '{future_train_end_time}' THEN '{past_train_end_time}'::TIMESTAMP"
    s += f" ELSE '{date_list[0]}'::TIMESTAMP END"
    return s

for dp_idx in range(10):
    date_pairs = [ date_pairs_all[2*dp_idx], date_pairs_all[2*dp_idx+1] ]
    print(date_pairs)
    params = {}
    params['pg_role'] = config["user"]
    params['schema'] = 'bias_working'
    experiment_hashes = ['a33cbdb3208b0df5f4286237a6dbcf8f']
    params['experiment_hashes'] = experiment_hashes
    if isinstance(date_pairs[0], str):
        date_pairs = [date_pairs]
    params['date_pairs'] = date_pairs
    params['date_weight_case_str'] = get_weight_case_str(date_weights)
    params['date_weight_past_train_end_time_case_str'] = get_weight_past_train_end_time_case_str(date_weights)
    params['list_sizes'] = [1000]
    params['demo_col'] = 'plevel'
    params['subsample'] = False
    params['bootstrap'] = False
    entity_demos='bias_working.entity_demos'
    engine=engine_donors.connect()
    validate_dates(params)
    if entity_demos.find('.') > -1:
        params['entity_demos'] = entity_demos
    elif entity_demos in ENTITY_DEMO_FILES.keys():
        params['entity_demos'] = create_entity_demos(engine, params, entity_demos)
    else:
        raise ValueError('Error: entity_demos must be either `schema.table_name` OR one of (%s)' % ', '.join(ENTITY_DEMO_FILES.keys()))

    sql = Template(open('recall_adjustment_pre.sql.tmpl', 'r').read()).render(**params)
    engine.execute(sql)
    engine.execute("COMMIT")
    
    
    sql = "SELECT %s, COUNT(*) AS num FROM %s GROUP BY 1 ORDER BY 2 DESC" % (params['demo_col'], params['entity_demos'])
    res = engine.execute(sql).fetchall()
    params['demo_values'] = [r[0] for r in res]
    params['demo_permutations'] = list(permutations(params['demo_values'], 2))

    
    # pre-calculate the results for all models, date pairs
    sql = Template(open('recall_adjustment-verbose.sql.tmpl', 'r').read()).render(**params)
    engine.execute(sql)
    engine.execute("COMMIT")
    
    
    engine.close()
    engine_donors.execute("""
        INSERT INTO bias_results.model_adjustment_results_plevel 
        SELECT * FROM bias_working.model_adjustment_results_plevel;
    """)
    
    engine_donors.execute("""
        INSERT INTO bias_results.composite_results_plevel 
        SELECT * FROM bias_working.composite_results_plevel;
    """)
    
    engine_donors.execute("""
        INSERT INTO bias_results.model_adjustment_group_k_plevel 
        SELECT * FROM bias_working.model_adjustment_group_k_plevel gkp WHERE (gkp.model_group_id, gkp.train_end_time, gkp.demo_value) NOT IN (SELECT model_group_id, train_end_time, demo_value FROM bias_results.model_adjustment_group_k_plevel)
    """)
    
    engine_donors.execute("""
        INSERT INTO bias_results.model_multi_adjustment_results_plevel
        SELECT * FROM bias_working.model_multi_adjustment_results_plevel;
    """)
    
    engine_donors.execute("COMMIT;")
