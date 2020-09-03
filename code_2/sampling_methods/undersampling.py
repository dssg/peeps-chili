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
from triage import create_engine
import pandas as pd
import shutil
from triage.component.catwalk.storage import ProjectStorage, ModelStorageEngine, MatrixStorageEngine

'''
Works as follows:
- Gets train matrices from an experiment id.
- Choose ratio
- Modifies the train matrix in certain way and inserts it into different dir structures.
- Undersampling bruh.
'''
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

def read_train_matrix(matrix_id, orig_data_dir, entity_to_attrib):
    matrix_data = pd.read_csv(os.path.join(orig_data_dir, str(matrix_id)+'.csv.gz'),
    compression = 'gzip', error_bad_lines=False)

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

    print("Number Missing="+str(missed))
    matrix_data['race_2way'] = race_col
    matrix_data = matrix_data[matrix_data['race_2way']!='MISSING']
    white_df = matrix_data[matrix_data['race_2way'] == 'White']
    non_white_df = matrix_data[matrix_data['race_2way'] == 'NonWhite']

    return white_df, non_white_df

def undersample(train_matrix_ids, orig_data_dir, dest_data_dir, ratio, N, entity_to_attrib):   
    for matrix_id in train_matrix_ids:  
        white_df, non_white_df = read_train_matrix(matrix_id, orig_data_dir, entity_to_attrib)
        
        if(len(non_white_df) > len(white_df)):
            larger_df = non_white_df
            smaller_df = white_df
        else:
            larger_df = white_df
            smaller_df = non_white_df
            
        #if ratio is 1:1, sample n_small from larger_df
        for i in range(N):
            if ratio * len(smaller_df) <= len(larger_df):
                sampled_larger_df = larger_df.sample(n=ratio*len(smaller_df))
                merged_df = pd.concat([smaller_df, sampled_larger_df])
                print(len(smaller_df), len(sampled_larger_df))
            else:
                # sample len(larger_df)/ratio from smaller_df
                sampled_smaller_df = smaller_df.sample(n = int(len(larger_df)/ratio))
                merged_df = pd.concat([sampled_smaller_df, larger_df])
                print(len(sampled_smaller_df), len(larger_df))
        
            us_dest_data_dir = os.path.join(dest_data_dir, 'undersampled_'+str(i), 'matrices')

            if not os.path.exists(us_dest_data_dir):
                os.makedirs(us_dest_data_dir)
        
            print("UNDERSAMPLING STAT")
            print(pd.value_counts(merged_df['race_2way']))
            print(len(merged_df[(merged_df['race_2way']=='White') & (merged_df['booking_view_warr_bw_1y']==0.0)]))
            print(len(merged_df[(merged_df['race_2way']=='White') & (merged_df['booking_view_warr_bw_1y']==1.0)]))
            print(len(merged_df[(merged_df['race_2way']=='NonWhite') & (merged_df['booking_view_warr_bw_1y']==0.0)]))
            print(len(merged_df[(merged_df['race_2way']=='NonWhite') & (merged_df['booking_view_warr_bw_1y']==1.0)]))
            print("="*30)

            filtered_df = merged_df.drop('race_2way', 1)
            filtered_df.to_csv(os.path.join(us_dest_data_dir, str(matrix_id)+".csv.gz"), compression='gzip', index=False)

def undersample_maintain_subgroup(train_matrix_ids, orig_data_dir, dest_data_dir, ratio, N, entity_to_attrib):
    for matrix_id in train_matrix_ids:
        white_df, non_white_df = read_train_matrix(matrix_id, orig_data_dir, entity_to_attrib)

        # PROCESS white_df
        white_0_df = white_df[white_df['booking_view_warr_bw_1y']==0.0]
        white_1_df = white_df[white_df['booking_view_warr_bw_1y']==1.0]

        non_white_0_df = non_white_df[non_white_df['booking_view_warr_bw_1y']==0.0]
        non_white_1_df = non_white_df[non_white_df['booking_view_warr_bw_1y']==1.0]


        for i in range(N):
            if(len(white_0_df) <  len(white_1_df)):
                # sample some from white_1
                sampled_white_1_df = white_1_df.sample(n=len(white_0_df))
                merged_white_df = pd.concat([white_0_df, sampled_white_1_df])
            else:
                sampled_white_0_df = white_0_df.sample(n=len(white_1_df))
                merged_white_df = pd.concat([sampled_white_0_df, white_1_df])

        
            # PROCESS non_white_df    
            if(len(non_white_0_df) < len(non_white_1_df)):
                # sample some from non_white_1
                sampled_non_white_1_df = non_white_1_df.sample(n=len(non_white_0_df))
                merged_non_white_df = pd.concat([non_white_0_df, sampled_non_white_1_df])
            else:
                sampled_non_white_0_df = non_white_0_df.sample(n=len(non_white_1_df))
                merged_non_white_df = pd.concat([sampled_non_white_0_df, non_white_1_df])

            us_dest_data_dir = os.path.join(dest_data_dir, 'undersampled_ratio2_frac_'+str(i), 'matrices')
            if not os.path.exists(us_dest_data_dir):
                os.makedirs(us_dest_data_dir)
        
            merged_df = pd.concat([merged_white_df, merged_non_white_df])
            print("UNDERSAMPLING RANDOM STAT")
            print(pd.value_counts(merged_df['race_2way']))
            print(len(merged_df[(merged_df['race_2way']=='White') & (merged_df['booking_view_warr_bw_1y']==0.0)]))
            print(len(merged_df[(merged_df['race_2way']=='White') & (merged_df['booking_view_warr_bw_1y']==1.0)]))
            print(len(merged_df[(merged_df['race_2way']=='NonWhite') & (merged_df['booking_view_warr_bw_1y']==0.0)]))
            print(len(merged_df[(merged_df['race_2way']=='NonWhite') & (merged_df['booking_view_warr_bw_1y']==1.0)]))
            print("="*30)
            
            filtered_df = merged_df.drop('race_2way', 1)
            filtered_df.to_csv(os.path.join(us_dest_data_dir, str(matrix_id)+".csv.gz"), compression='gzip', index=False)

def undersample_multiple_times(conn, orig_data_dir, dest_data_dir, experiment_hash, ratio, N, mode, entity_to_attrib):
    train_matrix_ids = get_train_matrices_ids(conn, experiment_hash)
    print("NUMBER OF MATRICES="+str(len(train_matrix_ids)))
    
    if(mode==1):
        undersample(train_matrix_ids, orig_data_dir, dest_data_dir, ratio, N, entity_to_attrib)
    if(mode==2):
        undersample_maintain_subgroup(train_matrix_ids, orig_data_dir, dest_data_dir, ratio, N, entity_to_attrib)

def copy_test_yaml(conn, orig_data_dir, dest_data_dir, experiment_hash, ratio, N):
    test_matrix_ids = get_test_matrices_ids(conn, experiment_hash)
    
    for matrix_id in test_matrix_ids:
        matrix_file_name = os.path.join(orig_data_dir, str(matrix_id)+'.csv.gz')
        for i in range(N):
            dest_file_name = os.path.join(dest_data_dir, 'undersampled_ratio2_frac_'+str(i), 'matrices', str(matrix_id)+'.csv.gz')
            print("Copying to:"+str(dest_file_name))
            shutil.copy(matrix_file_name, dest_file_name)

    print("Orig data dir="+str(os.path.join(orig_data_dir)))
    for in_file in os.listdir(os.path.join(orig_data_dir)):
        if(in_file.endswith("yaml")):
            print("Copying:"+str(os.path.join(orig_data_dir,in_file))+" TO "+str(os.path.join(dest_data_dir, 'undersampled_ratio2_frac_')))
            for i in range(N):
                shutil.copy(os.path.join(orig_data_dir, in_file), 
                os.path.join(dest_data_dir, 'undersampled_ratio2_frac_'+str(i), 'matrices', in_file))

if __name__ == '__main__':
    DATA_DIR = '/mnt/data/experiment_data/peeps'
    conn = connect('../../config')
    orig_data_dir = os.path.join(DATA_DIR, 'joco_original', 'matrices')
    dest_data_dir = os.path.join(DATA_DIR, 'joco_undersampled_race')
    experiment_hash = sys.argv[1]
    N = int(sys.argv[2])
    mode = int(sys.argv[3])
    ratio = 2
    
    print("Doing FOR:")
    print("\tExperiment Hash="+str(experiment_hash))
    print("\tN="+str(N))
    print("\tmode="+str(mode))
    print("="*30)
    
    entity_to_attrib = get_entities_attribute(conn)
    undersample_multiple_times(conn, orig_data_dir, dest_data_dir, experiment_hash, ratio, N, mode, entity_to_attrib)
    copy_test_yaml(conn, orig_data_dir, dest_data_dir, experiment_hash, ratio, N)