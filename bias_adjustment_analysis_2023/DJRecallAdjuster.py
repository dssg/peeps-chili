import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

import os
import warnings
import yaml
import sqlalchemy


from itertools import permutations
from jinja2 import Template
import dateparser
import string

from ohio.ext.numpy import pg_copy_to_table

try:
    import matplotlib.pyplot as plt
except (ImportError, RuntimeError) as e:
    print("matplotlib import error -- you are likely using the terminal, so plot() functions will not be available")
    pass




class RecallAdjuster(object):
    def __init__(
        self,
        engine,
        params,
        pause_phases=False, 
        exhaustive=False, 
        small_model_selection=False, 
        entity_analysis=False):
        """
        Arguments:
            engine: 
                An engine for a postgres database
            params:
                Dictionary with following properly defined
                pg_role:
                    Role to use in postgres
                schema:
                    Schema for table creation
                experiment_hashes:
                    A list of strings with triage experiments to include
                date_pairs:
                    A list of tuples of train_end_times (as strings). The first should be the date to use
                    to make adjustments and the second a future date for evaluation.
                list_sizes:
                    A list of integers, the sizes of lists to generate.
                entity_demos:
                    Either a table name (in format "schema.table_name") or "joco" to use
                    JoCo-specific code to create this table on the fly.
                demo_col:
                    Column name containing demographic data on which to make adjustments
                sample_weights:
                    Optional dictionary of demo values to weight for subsampling. Excluded demo values
                    will be included in their entirely with no sampling. Weights should be a value
                    between 0 and 1.0, reflecting the fraction of that demographic to include.
                decoupled_experiments:
                    Optional list of tuples of (experiment_hash, demo_value) that identify decoupled
                    experiments with models run using data only from each subgroup. Data from these
                    experiments will be used only to create a composite that allows EITHER the decoupled
                    or full models to be used for each subgroup. Multiple experiments can be specified
                    for a given demo_value, but all demo_values must be included.
                decoupled_entity_demos:
                    Optional "schema.table_name" for a separate entity_demos table to be used for the decoupled
                    experiments, for instance in cases where entity_ids may differ between modeling runs
                    such as is the case with JoCo matches. If specified, must be pre-computed.
                entity_demos:
                    e.g: {working_schema}.entity_demos
                weights:
                    Weighting scheme for multi adjustment as a list of fractions
                date_list:
                    List of all dates in increasing order
                min_separations:
                    The minimum time we must go back given a future_train_end_time to know we have that data and label set at prediction time of future_train_end_time. If not stated we assume 2
            pause_phases:
                True if you want a break after each phase requiring user input to continue
            exhaustive:
                Runs bias adjustment with group_k +- 50 on both sides to measure stability of results to adjustment thresholds
            small_model_selection:
                True if you want to use the model_adjustment_results_{demo_col} table to select only the best models for the next step of recall adjustment. 
            entity_analysis:
                Runs entity selection instead of recall adjustment
        """

        # store parameters
        self.engine = engine.connect()
        self.params = params
        
        self.params['date_weights'] = self.get_date_weights()
        self.params['date_weight_case_str'] = self.get_weight_case_str()
        self.params['date_weight_past_train_end_time_case_str'] = self.get_weight_past_train_end_time_case_str()
        self.params['single_model_str'] = self.get_single_model_str()
        

        # check consistency of date pairs
        self.validate_dates()

        # create a few temporary tables we'll need for calculations
        if small_model_selection:
            pre_file = "general/quick_adjustment_pre.sql.tmpl"
        else:
            pre_file = "general/recall_adjustment_pre.sql.tmpl"
        
        sql = Template(open(pre_file, 'r').read()).render(**self.params)
        self.engine.execute(sql)
        self.engine.execute("COMMIT")
        
        if pause_phases:
            input(f"Date Pair: {self.params['date_pairs']} pre sql done")

        entity_demos = self.params['entity_demos']
        if entity_demos.find('.') > -1:
            self.params['entity_demos'] = entity_demos
        else:
            raise ValueError('Error: entity_demos must be either `schema.table_name`')

        # calculate demo values for general use, ordered by frequency
        sql = "SELECT %s, COUNT(*) AS num FROM %s GROUP BY 1 ORDER BY 2 DESC" % (self.params['demo_col'], self.params['entity_demos'])
        res = self.engine.execute(sql).fetchall()
        self.params['demo_values'] = [r[0] for r in res]
        self.params['demo_permutations'] = list(permutations(self.params['demo_values'], 2))
        

        # pre-calculate the results for all models, date pairs
        self.params['tmp_rolling_recall_str'] = self.get_tmp_rolling_recall_str()        
        adjustment_file = 'general/recall_adjustment_verbose.sql.tmpl'
        if exhaustive:
            adjustment_file = "general/recall_adjustment_exhaustive.sql.tmpl"
            self.params['exhaustive_list'] = self.get_exhaustive_list()
        if entity_analysis:
            adjustment_file = "general/entity_analysis.sql.tmpl"
        sql = Template(open(adjustment_file, 'r').read()).render(**self.params)
        self.engine.execute(sql)
        self.engine.execute("COMMIT")
        
        if pause_phases:
            input(f"Date Pair: {self.params['date_pairs']} Adjustment Done")

        # store the results to dataframes for subsequent plotting and analysis
        sql = 'SELECT * FROM %s.model_adjustment_results_%s' % (self.params['schema'], self.params['demo_col'])
        self.adjustment_results = pd.read_sql(sql, self.engine)


        self.engine.close()

    
    def get_date_weights(self):
        weights = self.params['weights']
        date_list = self.params['date_list']
        min_separation = self.params.get('min_separations', 1)
        date_weights = {}
        i_lim = len(weights)
        assert i_lim >= min_separation
        for i, date in enumerate(date_list):
            if i < i_lim + min_separation - 1:
                date_weights[date] = {date_list[0]: 1.0, "past_train_end_time": date_list[0]}
            else:
                base = i - min_separation # No matter what latest time we can use is 2 before current month
                d = {}
                for j in range(len(weights)):
                    d[date_list[base - j]] = weights[j]
                d["past_train_end_time"] = date_list[base] # Should be the most recent date used to compute adjustment
                date_weights[date] = d
        return date_weights

    
    
    def get_weight_past_train_end_time_case_str(self):
        date_weights = self.params['date_weights']
        date_list = self.params['date_list']
        if len(date_weights) == 0:
            return f"'{date_list[0]}'::TIMESTAMP"
        s = "CASE"
        for future_train_end_time in date_weights:
            past_train_end_time = date_weights[future_train_end_time]["past_train_end_time"]
            s += f" WHEN future_train_end_time = '{future_train_end_time}' THEN '{past_train_end_time}'::TIMESTAMP"
        s += f" ELSE '{date_list[0]}'::TIMESTAMP END"
        return s
    
    def get_weight_case_str(self):
        date_weights = self.params['date_weights']
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
    
    def get_exhaustive_list(self):
        demo_values = self.params['demo_values']
        assert len(demo_values) == 2 # For now we will only do this with the even case
        low_num = self.params.get('exhaustive_low_frac', 0.25)
        high_num = self.params.get('exhaustive_high_frac', 0.75)
        n_steps = self.params.get('exhaustive_n_steps', 15)
        fracs = np.linspace(low_num, high_num, n_steps)
        d = []
        for i, frac in enumerate(fracs):
            case_str = f"CASE WHEN demo_value='{demo_values[0]}' then (list_size * {frac})::INT else (list_size * (1-{frac}))::INT END AS group_k"
            d.append((i, case_str))
        return d
    
    
    def get_tmp_rolling_recall_str(self):
        if self.params.get("coalesce", False):
            s = "COUNT(*) OVER w_roll AS num_demo_rolling, GREATEST(COUNT(label_value) OVER w_roll, 1) AS num_label_demo_rolling, COALESCE(SUM(label_value) OVER w_roll, 0) AS tp_demo_rolling, 1.0000*(COALESCE(SUM(label_value) OVER w_roll, 0))/(SUM(label_value) OVER w_all) AS recall_demo_rolling"
        else:
            s = "COUNT(*) OVER w_roll AS num_demo_rolling, SUM(label_value) OVER w_roll AS tp_demo_rolling, 1.0000*(SUM(label_value) OVER w_roll)/(SUM(label_value) OVER w_all) AS recall_demo_rolling"
        return s
            
    def get_single_model_str(self):
        model = self.params.get("single_model", None)
        if model is not None:
            return f" AND m.model_group_id = {model}"
        else:
            return ""
        
        

    
    def ensure_all_demos(self, check_demos):
        all_demos = set(self.params['demo_values'])
        check_demos = set(check_demos)
        if all_demos - check_demos:
            raise ValueError('Error: demo values not found in decoupled_experiments - %s' % (all_demos - check_demos))
        if check_demos - all_demos:
            raise ValueError('Error: demo values specified in decoupled_experiments not found in data - ' % (check_demos - all_demos))


    def validate_dates(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for past, future in self.params['date_pairs']:
                if dateparser.parse(past) > dateparser.parse(future):
                    raise ValueError('Error! Cannot validate on the past. %s should be no earlier than %s.' % (future, past))

                    
                    
def get_config(directory):
    database_config = None
    with open(directory+"/config.yaml") as f:
        database_config = yaml.full_load(f)
    if directory in []:
        database_config["coalesce"] = True
        
    if directory == "education_crowdfunding_replication":
        database_config['date_pairs_all'] = [('2011-03-01', '2011-03-01'), ('2011-03-01', '2011-07-01'), ('2011-05-01', '2011-05-01'), ('2011-05-01', '2011-09-01'), ('2011-07-01', '2011-07-01'), ('2011-07-01', '2011-11-01'), ('2011-09-01', '2011-09-01'), ('2011-09-01', '2012-01-01'), ('2011-11-01', '2011-11-01'), ('2011-11-01', '2012-03-01'), ('2012-01-01', '2012-01-01'), ('2012-01-01', '2012-05-01'), ('2012-03-01', '2012-03-01'), ('2012-03-01', '2012-07-01'), ('2012-05-01', '2012-05-01'), ('2012-05-01', '2012-09-01'), ('2012-07-01', '2012-07-01'), ('2012-07-01', '2012-11-01'), ('2012-09-01', '2012-09-01'), ('2012-09-01', '2013-01-01')]
    return database_config
    
                   
    
def ra_procedure(directory, weights=[0.99, 0.01], exhaustive_n_steps=15, alternate_save_names=[], engine_donors=None, config=None, pause_phases=False, exhaustive=False, small_model_selection=False, single_model=False, entity_analysis=False):
    if single_model:
        small_model_selection = True
    database_config = get_config(directory)
    database = database_config['database']
    demo_col =  database_config['demo_col']
    working_schema = database_config['working_schema']
    results_schema = database_config['results_schema']
    list_size = database_config['list_size']
    date_list = database_config['date_list']
    experiment_hashes = database_config['experiment_hashes']
    date_pairs_all = database_config.get('date_pairs_all', None)
    coalesce = database_config.get('coalesce', False)
    min_separations = database_config.get('min_separations', 1)
    if single_model or entity_analysis:
        single_model = database_config['single_model']
    else:
        single_model = None
    if engine_donors is None or config is None:
        with open('../config/db_default_profile.yaml') as fd:
            config = yaml.full_load(fd)
            dburl = sqlalchemy.engine.url.URL.create(
                "postgresql",
                host=config["host"],
                username=config["user"],
                database=database,
                password=config["pass"],
                port=config["port"],
            )
            engine_donors = sqlalchemy.create_engine(dburl, poolclass=sqlalchemy.pool.QueuePool)


    if date_pairs_all is None:
        date_pairs_all = []
        for i, d in enumerate(date_list[:-1]):
            date_pairs_all.append((date_list[i], date_list[i]))
            date_pairs_all.append((date_list[i], date_list[i+1]))
            
    common_params = {"pg_role": config["user"], "schema": working_schema, "experiment_hashes": experiment_hashes, 'demo_col': demo_col, "subsample": False, "bootstrap": False, "entity_demos":f'{working_schema}.entity_demos', "list_sizes": [list_size], "date_list": date_list, "min_separations": min_separations, "exhaustive_n_steps": exhaustive_n_steps, 'coalesce': coalesce, 'single_model': single_model}

    if not entity_analysis:    
        engine_donors.execute(f'TRUNCATE TABLE {results_schema}.model_adjustment_results_{demo_col};')
        engine_donors.execute(f'TRUNCATE TABLE {working_schema}.model_adjustment_group_k_{demo_col};')
        if exhaustive:
            for index in range(exhaustive_n_steps):
                engine_donors.execute(f"DROP TABLE IF EXISTS {results_schema}.exhaustive_{index};")
                engine_donors.execute(f"CREATE TABLE {results_schema}.exhaustive_{index} AS SELECT * FROM {results_schema}.model_adjustment_results_{demo_col};")
                engine_donors.execute(f'TRUNCATE TABLE {results_schema}.exhaustive_{index};')

        engine_donors.execute('COMMIT;')

        for dp_idx in range(0, len(date_pairs_all), 2):
            date_pairs = [ date_pairs_all[dp_idx], date_pairs_all[dp_idx+1] ]
            print(date_pairs)
            params = common_params.copy()
            if isinstance(date_pairs[0], str):
                date_pairs = [date_pairs]
            params['date_pairs'] = date_pairs
            params['weights'] = weights


            engine=engine_donors
            ra = RecallAdjuster(engine=engine, params=params, pause_phases=pause_phases, exhaustive=exhaustive, small_model_selection=small_model_selection)

            engine_donors.execute(f"""
                INSERT INTO {results_schema}.model_adjustment_results_{demo_col} 
                SELECT * FROM {working_schema}.model_adjustment_results_{demo_col};
            """)

            engine_donors.execute(f"""
                INSERT INTO {results_schema}.model_adjustment_group_k_{demo_col} 
                SELECT * FROM {working_schema}.model_adjustment_group_k_{demo_col} gkp WHERE (gkp.model_group_id, gkp.train_end_time, gkp.demo_value, gkp.group_k) NOT IN (SELECT * FROM {results_schema}.model_adjustment_group_k_{demo_col})
            """)

            if exhaustive:
                for index in range(exhaustive_n_steps):
                    engine_donors.execute(f"""
                        INSERT INTO {results_schema}.exhaustive_{index} 
                        SELECT * FROM {working_schema}.exhaustive_{index};
                    """)

            engine_donors.execute("COMMIT;")

        for save_name in alternate_save_names:
            schema = params['schema'] 
            demo_col = params["demo_col"]
            sql = f"DROP TABLE IF EXISTS {results_schema}.{save_name}; CREATE TABLE {results_schema}.{save_name} AS SELECT * FROM {results_schema}.model_adjustment_results_{demo_col};"
            engine_donors.execute(sql)
            engine_donors.execute("COMMIT;")
    else:
        engine_donors.execute(f"drop table if exists {working_schema}.entity_distribution;")
        sql = f"create table {working_schema}.entity_distribution as (select '2012-09-01 00:00:00.000'::TIMESTAMP as train_end_time, 1 as model_rank, 1 as rn_demo, 'n' as demo_value, 0.1 as recall_demo_rolling, 0.1 as precision_demo_rolling, 0.1 as score from {working_schema}.tmp_bias_models);"
        engine_donors.execute(sql)
        engine_donors.execute(f"truncate table {working_schema}.entity_distribution;")
        

        engine_donors.execute(f"drop table if exists {working_schema}.entity_delta_distribution;")
        sql = f"create table {working_schema}.entity_delta_distribution as (select '2012-09-01 00:00:00.000'::TIMESTAMP as train_end_time, 'n' as demo_value, 1 as base_group_k, 1 as base_model_rank, 0.1 as base_recall_demo_rolling, 0.1 as base_precision_demo_rolling, 0.1 as base_score, 1 as adj_group_k, 1 as adj_model_rank, 0.1 as adj_recall_demo_rolling, 0.1 as adj_precision_demo_rolling, 0.1 as adj_score, 1 as multi_adj_group_k, 1 as multi_adj_model_rank, 0.1 as multi_adj_recall_demo_rolling, 0.1 as multi_adj_precision_demo_rolling, 0.1 as multi_adj_score from {working_schema}.tmp_bias_models);"
        engine_donors.execute(sql)
        engine_donors.execute(f"truncate table {working_schema}.entity_delta_distribution")
        
        engine_donors.execute("COMMIT")
        for dp_idx in range(0, len(date_pairs_all), 2):
            date_pairs = [ date_pairs_all[dp_idx], date_pairs_all[dp_idx+1] ]
            print(date_pairs)
            params = common_params.copy()
            if isinstance(date_pairs[0], str):
                date_pairs = [date_pairs]
            params['date_pairs'] = date_pairs
            params['weights'] = weights


            
            engine=engine_donors
            ra = RecallAdjuster(engine=engine, params=params, pause_phases=pause_phases, small_model_selection=True, entity_analysis=True)
            engine_donors.execute("COMMIT")

        
def multi_weight_ra_procedure(directory, small_model_selection=False):
    w = 0.99
    print(f"Procedure with weights: {w}")
    ra_procedure(directory, weights=[w, 1-w], alternate_save_names=["save_res_a"], small_model_selection=small_model_selection)
    w = 0.9
    print(f"Procedure with weights: {w}")
    ra_procedure(directory, weights=[w, 1-w], alternate_save_names=["save_res_b"], small_model_selection=small_model_selection)
    w = 0.8
    print(f"Procedure with weights: {w}")    
    ra_procedure(directory, weights=[w, 1-w], alternate_save_names=["save_res_c"], small_model_selection=small_model_selection)
    w = 0.7
    print(f"Procedure with weights: {w}")
    ra_procedure(directory, weights=[w, 1-w], alternate_save_names=["save_res_d"], small_model_selection=small_model_selection)
    w = 0.6
    print(f"Procedure with weights: {w}")
    ra_procedure(directory, weights=[w, 1-w], alternate_save_names=["save_res_e"], small_model_selection=small_model_selection)
    w = 0.5
    print(f"Procedure with weights: {w}")
    ra_procedure(directory, weights=[w, 1-w], alternate_save_names=["save_res_f"], small_model_selection=small_model_selection)
    w = 0.4
    print(f"Procedure with weights: {w}")
    ra_procedure(directory, weights=[w, 1-w], alternate_save_names=["save_res_g"], small_model_selection=small_model_selection)
    w = 0.3
    print(f"Procedure with weights: {w}")
    ra_procedure(directory, weights=[w, 1-w], alternate_save_names=["save_res_h"], small_model_selection=small_model_selection)
    w = 0.2
    print(f"Procedure with weights: {w}")
    ra_procedure(directory, weights=[w, 1-w], alternate_save_names=["save_res_i"], small_model_selection=small_model_selection)
    w = 0.1
    print(f"Procedure with weights: {w}")
    ra_procedure(directory, weights=[w, 1-w], alternate_save_names=["save_res_j"], small_model_selection=small_model_selection)
    w = 0.01
    print(f"Procedure with weights: {w}")
    ra_procedure(directory, weights=[w, 1-w], alternate_save_names=["save_res_k"], small_model_selection=small_model_selection)
    
        

if __name__ == "__main__":
    ra_procedure(directory, weights=[1, 0], pause_phases=False, entity_selection=False, small_model_selection=False)
    