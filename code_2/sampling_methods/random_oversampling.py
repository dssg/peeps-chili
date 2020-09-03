import os
import sys
project_path = '../..'
sys.path.append(project_path)
import getpass
import ntpath
import logging
import time
logging.basicConfig(level=logging.DEBUG, filename="LOG_Triage.debug", filemode='w')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
from datetime import datetime
import yaml
import pandas as pd
import shutil
from sqlalchemy import create_engine
import numpy as np

def read_config_file(config_file):
    config = None
    try:
        with open (config_file, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(e)
        print('Error reading the config file')
    return config

def connect(cred_folder):
    cred_file = os.path.join(cred_folder, 'joco_db_profile.yaml')
    db = read_config_file(cred_file)

    sql_engine = create_engine(
        'postgresql+psycopg2://%s:%s@%s:%i/%s'%(
            db['user'],
            db['pass'],
            db['host'],
            db['port'],
            db['db']
        )
    )
    return sql_engine

def get_entities_attribute(conn):
    entity_to_attrib = {}
    query = """
    SELECT entity_id, race_2way 
    FROM hemank_bias_alternatives.currmatch_entity_demos
    """
    df = pd.read_sql(query, conn)
    entities = df['entity_id'].values
    races = df['race_2way'].values
    
    for i in range(len(entities)):
        entity_id = entities[i]
        race_info = races[i]

        entity_to_attrib[entity_id] = race_info

    return entity_to_attrib

def get_test_matrices_ids(conn, experiment_id):
    query = """
    SELECT matrix_uuid from model_metadata.matrices where matrix_type='test' AND
    built_by_experiment = '%s'
    """%(str(experiment_id))

    df = pd.read_sql(query, conn)
    test_matrices = df['matrix_uuid'].values

    return test_matrices

def get_train_matrices_ids(conn, experiment_id):
    query = """
    SELECT matrix_uuid from model_metadata.matrices where matrix_type='train' AND
    built_by_experiment = '%s'
    """%(str(experiment_id))

    df = pd.read_sql(query, conn)
    train_matrices = df['matrix_uuid'].values

    return train_matrices

def read_train_matrix(matrix_id, orig_data_dir):
    matrix_data = pd.read_csv(os.path.join(orig_data_dir, str(matrix_id)+'.csv.gz'),
    compression = 'gzip', error_bad_lines=False)

    white_df = matrix_data[matrix_data['protected_demographics_entity_id_all_race_W_max']==1]
    non_white_df = matrix_data[matrix_data['protected_demographics_entity_id_all_race_W_max']==0]

    return white_df, non_white_df

def random_sample(df_one, df_two):
    if(len(df_one) > len(df_two)):
        majority_df = df_one
        minority_df = df_two
    else:
        majority_df = df_two
        minority_df = df_one
    
    diff = len(majority_df) - len(minority_df)

    sampled_minority = minority_df.sample(n=diff, replace=True)
    concat_df = pd.concat([majority_df, minority_df, sampled_minority])

    #for col in concat_df.columns:
    #    print(col)
    
    #print(concat_df.column)
    #concat_df['entity_id'] = range(len(concat_df))
    #min_date = min(sampled_minority['as_of_date'])
    #concat_df['as_of_date'] = pd.date_range(min_date, periods=len(concat_df), freq='D')
    return concat_df

def random_oversample(train_matrix_ids, orig_data_dir, dest_data_dir, ratio, N, entity_to_attrib):
    # Make Whites more equal to Non-Whites.
    # make sure that label is discrete i.e. 0 or 1.    
    for matrix_id in train_matrix_ids:  
        matrix_data = pd.read_csv(os.path.join(orig_data_dir, str(matrix_id)+'.csv.gz'), 
        compression='gzip', error_bad_lines=False)

        entities = matrix_data['entity_id'].values
        race_col = []
        missed = 0
        for i in range(len(entities)):
            entity_id = entities[i]
            try:
                race_info = entity_to_attrib[entity_id]
                race_col.append(race_info)
            except KeyError as e:
                missed = missed + 1
                race_col.append('MISSING')

        print("Number MISSED="+str(missed))
        matrix_data['race_2way'] = race_col
        matrix_data = matrix_data[matrix_data['race_2way']!='MISSING']
        white_data = matrix_data[matrix_data['race_2way']=='White']
        non_white_data = matrix_data[matrix_data['race_2way']=='NonWhite']

        for i in range(N):
            balanced_df = random_sample(white_data, non_white_data)
            us_dest_data_dir = os.path.join(dest_data_dir, 'oversampled_random_'+str(i), 'matrices')
            
            if not os.path.exists(us_dest_data_dir):
                os.makedirs(us_dest_data_dir)

            print("OVERSAMPLED STAT:")
            print(
                len(balanced_df[(balanced_df['race_2way'] == 'White') & (balanced_df['booking_view_warr_bw_1y'] == 0.0)]),
                len(balanced_df[(balanced_df['race_2way'] == 'NonWhite') & (balanced_df['booking_view_warr_bw_1y'] == 0.0)]),
                len(balanced_df[(balanced_df['race_2way'] == 'White') & (balanced_df['booking_view_warr_bw_1y'] == 1.0)]),
                len(balanced_df[(balanced_df['race_2way'] == 'NonWhite') & (balanced_df['booking_view_warr_bw_1y'] == 1.0)]))
            print("="*30)

            balanced_df['entity_id'] = range(len(balanced_df))

            print(pd.value_counts(balanced_df['entity_id'].values))
            filtered_df = balanced_df.drop('race_2way', 1)
            filtered_df.to_csv(os.path.join(us_dest_data_dir, str(matrix_id)+".csv.gz"), 
            compression='gzip', index=False)

def random_oversample_maintain_group(train_matrix_ids, orig_data_dir, dest_data_dir, ratio, N, entity_to_attrib):
    mid=0
    for matrix_id in train_matrix_ids:
        print("MATRIX:"+str(mid)+"/"+str(len(train_matrix_ids)))
        mid = mid + 1
        matrix_data = pd.read_csv(os.path.join(orig_data_dir, str(matrix_id)+'.csv.gz'), 
        compression='gzip', error_bad_lines=False)
        
        entities = matrix_data['entity_id'].values
        race_col = []
        missed = 0
        for i in range(len(entities)):
            entity_id = entities[i]
            try:
                race_info = entity_to_attrib[entity_id]
                race_col.append(race_info)
            except KeyError as e:
                missed+=1
                race_col.append("MISSING")

        print("Number MISSED="+str(missed))

        matrix_data['race_2way'] = race_col
        matrix_data = matrix_data[matrix_data['race_2way']!='MISSING']
        white_data = matrix_data[matrix_data['race_2way'] == 'White']
        non_white_data = matrix_data[matrix_data['race_2way'] == 'NonWhite']

        # Randomly oversample within white
        for i in range(N):
            white_0 = white_data[white_data['booking_view_warr_bw_1y']==0.0]
            white_1 = white_data[white_data['booking_view_warr_bw_1y']==1.0]
            white_balanced_df = random_sample(white_0, white_1)

            non_white_0 = non_white_data[non_white_data['booking_view_warr_bw_1y']==0.0]
            non_white_1 = non_white_data[non_white_data['booking_view_warr_bw_1y']==1.0]
            non_white_balanced_df = random_sample(non_white_0, non_white_1)

            merged_df = pd.concat([white_balanced_df, non_white_balanced_df])

            us_dest_data_dir = os.path.join(dest_data_dir, 'oversampled_random_frac_'+str(i), 'matrices')
            if not os.path.exists(us_dest_data_dir):
                os.makedirs(us_dest_data_dir)

            merged_df['entity_id'] = range(len(merged_df))

            print("OVERSAMPLED STAT:")
            print(
                len(merged_df[(merged_df['race_2way'] == 'White') & (merged_df['booking_view_warr_bw_1y'] == 0.0)]),
                len(merged_df[(merged_df['race_2way'] == 'NonWhite') & (merged_df['booking_view_warr_bw_1y'] == 0.0)]),
                len(merged_df[(merged_df['race_2way'] == 'White') & (merged_df['booking_view_warr_bw_1y'] == 1.0)]),
                len(merged_df[(merged_df['race_2way'] == 'NonWhite') & (merged_df['booking_view_warr_bw_1y'] == 1.0)]))
            print("="*30)
            
            print(pd.value_counts(merged_df['entity_id'].values))
            filtered_df = merged_df.drop('race_2way', 1)
            filtered_df.to_csv(os.path.join(us_dest_data_dir, str(matrix_id)+".csv.gz"), compression='gzip', index=False)

def random_oversample_multiple_times(conn, orig_data_dir, dest_data_dir, experiment_hash, ratio, N, mode, entity_to_attrib):
    train_matrix_ids = get_train_matrices_ids(conn, experiment_hash)
    print("Number of train matrices obtained="+str(len(train_matrix_ids)))
    if(mode==1):
        random_oversample(train_matrix_ids, orig_data_dir, dest_data_dir, ratio, N, entity_to_attrib)
    if(mode==2):
        random_oversample_maintain_group(train_matrix_ids, orig_data_dir, dest_data_dir, ratio, N, entity_to_attrib)

def copy_test_yaml(conn, orig_data_dir, dest_data_dir, experiment_hash, ratio, N):
    test_matrix_ids = get_test_matrices_ids(conn, experiment_hash)
    for matrix_id in test_matrix_ids:
        matrix_file_name = os.path.join(orig_data_dir, str(matrix_id)+'.csv.gz')
        for i in range(N):
            dest_file_name = os.path.join(dest_data_dir, 'oversampled_random_frac_'+str(i), 'matrices', str(matrix_id)+'.csv.gz')
            print("Copying to:"+str(dest_file_name))
            shutil.copy(matrix_file_name, dest_file_name)

    print("Orig data dir="+str(os.path.join(orig_data_dir)))
    for in_file in os.listdir(os.path.join(orig_data_dir)):
        if(in_file.endswith("yaml")):
            for i in range(N):
                print("Copying:"+str(os.path.join(orig_data_dir,in_file))+" TO "+str(os.path.join(dest_data_dir, 'oversampled_random_frac_'+str(i))))
                shutil.copy(os.path.join(orig_data_dir, in_file), 
                os.path.join(dest_data_dir, 'oversampled_random_frac_'+str(i), 'matrices', in_file))

if __name__ == '__main__':
    DATA_DIR = '/mnt/data/experiment_data/peeps'
    conn = connect('../../config')
    orig_data_dir = os.path.join(DATA_DIR, 'joco_original', 'matrices')
    dest_data_dir = os.path.join(DATA_DIR, 'joco_oversampled_race2')
    
    experiment_hash = sys.argv[1]
    ratio = 1
    N = int(sys.argv[2])
    mode = int(sys.argv[3])
    
    entity_to_attrib = get_entities_attribute(conn)
    random_oversample_multiple_times(conn, orig_data_dir, dest_data_dir, experiment_hash, ratio, N, mode, entity_to_attrib)
    copy_test_yaml(conn, orig_data_dir, dest_data_dir, experiment_hash, ratio, N)