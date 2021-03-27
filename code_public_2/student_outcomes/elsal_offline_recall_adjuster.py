import os
import yaml
import pandas as pd
import sqlalchemy
import datetime
from dateutil.relativedelta import relativedelta
import time
import sys
import SAVE_RecallAdjuster as sra
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('error.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)



def connect(poolclass=sqlalchemy.pool.QueuePool):
    with open(os.path.join('../..','config', 'elsal_db_profile2.yaml')) as fd:
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

def get_model_groups(conn, experiment_ids):
    experiment_hashes = "','".join([str(x) for x in experiment_ids])
    experiment_hashes = "'"+experiment_hashes+"'"

    '''
    query = """
    select distinct(model_id), model_group_id
    from test_results.predictions 
    join triage_metadata.models mm using (model_id)
    where mm.built_by_experiment IN
    (
        %s
    )
    -- where model_group_id NOT IN (82,83,84,85,76,77,78,79,80,96,101,102,104,--105,106,107,108,109,110,116,122,123,124,125,126,127,128,133,134,136)
    """%(str(experiment_hashes))
    '''

    query = """
    select distinct(model_id), model_group_id
    from triage_metadata.models mm
    where mm.built_by_experiment IN
    ('%s', '%s')
    """%(str(experiment_ids[0]), str(experiment_ids[1]))
    
    df = pd.read_sql(query, conn)
    model_groups = df['model_group_id'].values
    
    return model_groups

def run_RA(date_pairs, model_groups, experiment_ids):
    start_time = time.time()
    print("Running Recall Adjuster with ")
    print("[ListSizes]:"+str([10000]))
    print("[ModelGroups]:"+str(model_groups))
    
    myRA = sra.RecallAdjuster(
        engine=conn,
        pg_role='postgres',
        schema='bias_analysis_v2_over_1_snop_orig',
        experiment_hashes=experiment_ids,
        date_pairs=date_pairs,
        list_sizes=[10000],
        entity_demos='bias_analysis_1year.entity_demos',
        demo_col='ovg',
        model_group_ids = model_groups,
        #decoupled_experiments = [('98cd54f7a8d866f3ac5f3894930d4556', '1'),('b06a12f77b7d0124c8c9e1edb27a60d0', '0')],

        #decoupled_experiments = [('98cd54f7a8d866f3ac5f3894930d4556', '1'), 
        #('63813d1fa09f83634210368538658694', '1'), 
        #('a3a778c094e0d943e96aa42e5220883f', '0'), 
        #('b06a12f77b7d0124c8c9e1edb27a60d0', '0')],
        dataset = ''
    )
    print("Time Taken="+str(time.time() - start_time))

if __name__ == "__main__":
    conn = connect()
    base_date = '2016-01-01'
    date_pairs = get_date_pairs(base_date, 6)

    '''
    experiment_ids = ['6733f3d6d9cf969163a8b636887c034e','22025a3cee46e30a983eda6928dc7e4c', '0564309a2648ab74f5c3fb4a205b026c',
    'd5b3552ddbc50294c9cfb79eb86182c0', '2eec3bb1bd8399b6af4683db4ef2c13b',
    '99f2aefab42eb3ecacf9c1d7a00ae627', 'a173625fa8141d745c8d06255eba4128',
    '3df4bab345bef7d6a12c3dec2b327dc6', '8997a251467002929fa7ddfe9a60a87b',
    '8eb6381ff72d182db994dae79e1a80d8', '8da0f5b2a3d55325bf90fe43bdb7b8b8',
    '0b62877e44ad9f9ef60fe22fdf3f0106', '708922dae6bf77ec3ed1869e77c0109b',
    '253710859ff3661bdaa2bedb5951164d']
    '''
    #experiment_ids = ['0237446c95f081a55dbac741a6043a74']
    #experiment_ids = ['cfe45b3126ce5ad534eb081d0da87c69']
    #experiment_ids = ['a67180209b65fa260bea41b47936c427']
    #experiment_ids = ['7f3d52ec76ff1c007753a32af821f97c']
    #experiment_ids = ['af9b55f87be82752250256f1d17e2978'] #under_orig_50_50
    #experiment_ids = ['e45d51f6d15d52f4e44e7021ce26c00a'] #under_1_50_50
    
    #experiment_ids = ['e68507361fa1c6dbeb1bc40e8c02d003'] #over_1_orig_orig
    #experiment_ids = ['4c02ae8079370278e72c1ab1baa102d7'] #over_1_snop_orig
    #experiment_ids = ['e7377180dfc2b31df7cf5e3b2518346d'] #over_orig_50_50
    #experiment_ids = ['9c430ba23e68274078a518e261794214'] #over_orig_50_orig
    #experiment_ids = ['73cdca3cee1f1b91ef974c9ee4c7231b'] #over_orig_snop_orig
    #experiment_ids = ['39a8ce8f1a99004d0112eeb29e78ccf5'] #over_1_50_50
    #experiment_ids = ['4850440471012dba6131d727a2bbc2e2'] # no_protected

    #experiment_ids = ['a1316f404aecc9df9e3c5264b32770f8'] #original for decoupled
    #decoupled_1_id = ['98cd54f7a8d866f3ac5f3894930d4556'] #overage1
    #decoupled_2_id = ['b06a12f77b7d0124c8c9e1edb27a60d0'] #overage0

    #experiment_ids = ['a1316f404aecc9df9e3c5264b32770f8', '290f77e278ed5b3ef07ad7e0f33fafc0'] #original
    #experiment_ids = ['0237446c95f081a55dbac741a6043a74', 'ac8387f4233138c7ac79528180ca0494'] #under_1_original_original
    #experiment_ids = ['cfe45b3126ce5ad534eb081d0da87c69', 'a157c48e5f588286b98d3129fcefc935'] #under_original_50_original
    #experiment_ids = ['a67180209b65fa260bea41b47936c427', '8c86b27e73efcc7c3c5a4d18105dd1d9'] #under_original_snop_original
    #experiment_ids = ['af9b55f87be82752250256f1d17e2978', '363de656751d98eb454c6d7598714110'] #under_original_50_50
    #experiment_ids  = ['e45d51f6d15d52f4e44e7021ce26c00a','732aa93f18c49c084469d07b2efe2f62']   #under_1_50_50
    #experiment_ids  = ['7f3d52ec76ff1c007753a32af821f97c','eee27ec2bc734f0cc7d99e1d007e0bed']   #under_1_snop_original

    #experiment_ids  = ['e68507361fa1c6dbeb1bc40e8c02d003','b7e81b902e04ebf2a5124b7706956543']   #over_1_original_original

    #experiment_ids = ['9c430ba23e68274078a518e261794214','5f40d36175ae546ecada8863984fd11e']    #over_original_50_original

    #experiment_ids = ['73cdca3cee1f1b91ef974c9ee4c7231b','6c62194e5d9b5a6353265a439eb57abc']    #over_original_snop_original

    #experiment_ids = ['e7377180dfc2b31df7cf5e3b2518346d','435f3e21e6f3fdb955ae41e409aaa040']    #over_original_50_50

    #experiment_ids = ['39a8ce8f1a99004d0112eeb29e78ccf5','e82ada1518d6ebc3ce3b93c57e84bb53']    #over_1_50_50

    experiment_ids = ['4c02ae8079370278e72c1ab1baa102d7','b4e0da3f19e7457b5d42b9c6d5d0f082']    #over_1_snop_original

    '''
    Using this only for getting relevant model groups --- which are as following:::
    1. 4c02ae8079370278e72c1ab1baa102d7 [BASIC MODELS]
    2. b4e0da3f19e7457b5d42b9c6d5d0f082 [RANDOM FOREST]
    '''
    mg_experiment_ids = ['4c02ae8079370278e72c1ab1baa102d7', 'b4e0da3f19e7457b5d42b9c6d5d0f082'] 
    
    model_groups = get_model_groups(conn, mg_experiment_ids)
    print("Model Groups Obtained:"+str(model_groups)+","+str(len(set(model_groups))))
    try:
        run_RA(date_pairs, model_groups, experiment_ids)
    except Exception as e:
        logger.error(str(e), exc_info=True)