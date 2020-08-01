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
from imblearn.over_sampling import SMOTE
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

def smote_oversample(train_matrix_ids, orig_data_dir, dest_data_dir, ratio, N):
    # smote_oversample is oversampling on just the race.
    # Make Whites more equal to Non-Whites.
    # make sure that label is discrete i.e. 0 or 1.
    for matrix_id in train_matrix_ids:  
        matrix_data = pd.read_csv(os.path.join(orig_data_dir, str(matrix_id)+'.csv.gz'), 
        compression='gzip', error_bad_lines=False)

        all_features = list(matrix_data.columns)
        x_features = all_features
        y_features = 'protected_demographics_entity_id_all_race_W_max'
        x_features.remove('entity_id')
        x_features.remove('as_of_date')
        x_features.remove('protected_demographics_entity_id_all_race_W_max')
        X = matrix_data[x_features]
        y = matrix_data[y_features]
        
        for i in range(N):
            X_resampled, y_resampled = SMOTE().fit_resample(X, y)
        
            entity_ids = np.arange(len(X_resampled))
            as_of_dates = max(matrix_data['as_of_date'])
            X_resampled['entity_id'] = entity_ids
            X_resampled['as_of_date'] = as_of_dates        

            resampled_df = X_resampled
            resampled_df['protected_demographics_entity_id_all_race_W_max'] = y_resampled
        
            resampled_df['booking_view_warr_bw_1y'][resampled_df['booking_view_warr_bw_1y']>0.5] = 1.0
            resampled_df['booking_view_warr_bw_1y'][resampled_df['booking_view_warr_bw_1y']<=0.5] = 0.0

            print(pd.value_counts(resampled_df['protected_demographics_entity_id_all_race_W_max']))
            print(pd.value_counts(resampled_df['booking_view_warr_bw_1y']))
        
            us_dest_data_dir = os.path.join(dest_data_dir, 'oversampled_'+str(i), 'matrices')
            if not os.path.exists(us_dest_data_dir):
                os.makedirs(us_dest_data_dir)

            resampled_df.to_csv(os.path.join(us_dest_data_dir, str(matrix_id)+".csv.gz"), 
            compression='gzip', index=False)

def smote_oversample_maintain_group(train_matrix_ids, orig_data_dir, dest_data_dir, ratio, N):
    # smote_oversample is oversampling on just the race.
    # Make Whites more equal to Non-Whites.
    # make sure that label is discrete i.e. 0 or 1.
    mid=0
    for matrix_id in train_matrix_ids:
        print("MATRIX:"+str(mid)+"/"+str(len(train_matrix_ids)))
        mid = mid + 1
        matrix_data = pd.read_csv(os.path.join(orig_data_dir, str(matrix_id)+'.csv.gz'), 
        compression='gzip', error_bad_lines=False)
        
        white_df = matrix_data[matrix_data['protected_demographics_entity_id_all_race_W_max'] == 1]
        non_white_df = matrix_data[matrix_data['protected_demographics_entity_id_all_race_W_max'] == 0]

        print("Initial:"+str(len(white_df))+","+str(len(non_white_df)))

        all_features = list(matrix_data.columns)
        x_features = all_features
        y_features = 'booking_view_warr_bw_1y'
        x_features.remove('entity_id')
        x_features.remove('as_of_date')
        x_features.remove('protected_demographics_entity_id_all_race_W_max')
        
        white_X = white_df[x_features]
        white_y = white_df[y_features]

        non_white_X = non_white_df[x_features]
        non_white_y = non_white_df[y_features]

        for i in range(N):
            print("ROUND:"+str(i))
            white_X_resampled, white_y_resampled = SMOTE().fit_resample(white_X, white_y)

            entity_ids = np.arange(len(white_X_resampled))
            as_of_dates = max(matrix_data['as_of_date'])
            white_X_resampled['entity_id'] = entity_ids
            white_X_resampled['as_of_date'] = as_of_dates
            white_X_resampled['protected_demographics_entity_id_all_race_W_max'] = np.ones(len(white_X_resampled))
            white_resampled_df = white_X_resampled
            white_resampled_df['booking_view_warr_bw_1y'] = white_y_resampled
            
            print(len(white_X), len(white_X_resampled))

            non_white_X_resampled, non_white_y_resampled = SMOTE().fit_resample(non_white_X, non_white_y)
            entity_ids = np.arange(len(non_white_X_resampled))
            as_of_dates = max(matrix_data['as_of_date'])
            non_white_X_resampled['entity_id'] = entity_ids
            non_white_X_resampled['as_of_date'] = as_of_dates
            non_white_X_resampled['protected_demographics_entity_id_all_race_W_max'] = np.zeros(len(non_white_X_resampled))
            non_white_resampled_df = non_white_X_resampled
            non_white_resampled_df['booking_view_warr_bw_1y'] = non_white_y_resampled

            print(len(non_white_X), len(non_white_X_resampled))
            
            us_dest_data_dir = os.path.join(dest_data_dir, 'oversampled_frac_'+str(i), 'matrices')
            if not os.path.exists(us_dest_data_dir):
                os.makedirs(us_dest_data_dir)
        
            merged_df = pd.concat([white_resampled_df, non_white_resampled_df])
            print("stat")
            print(
                len(merged_df[(merged_df['protected_demographics_entity_id_all_race_W_max']==1) & (merged_df['booking_view_warr_bw_1y'] == 0.0)]),
                len(merged_df[(merged_df['protected_demographics_entity_id_all_race_W_max']==1) & (merged_df['booking_view_warr_bw_1y'] == 1.0)]),
                len(merged_df[(merged_df['protected_demographics_entity_id_all_race_W_max']==0) & (merged_df['booking_view_warr_bw_1y'] == 0.0)]),
                len(merged_df[(merged_df['protected_demographics_entity_id_all_race_W_max']==0) & (merged_df['booking_view_warr_bw_1y'] == 1.0)]))
            merged_df.to_csv(os.path.join(us_dest_data_dir, str(matrix_id)+".csv.gz"), compression='gzip', index=False)

def oversample_multiple_times(conn, orig_data_dir, dest_data_dir, experiment_hash, ratio, N, mode):
    train_matrix_ids = get_train_matrices_ids(conn, experiment_hash)
    print("Number of train matrices obtained="+str(len(train_matrix_ids)))
    if(mode==1):
        smote_oversample(train_matrix_ids, orig_data_dir, dest_data_dir, ratio, N)
    if(mode==2):
        smote_oversample_maintain_group(train_matrix_ids, orig_data_dir, dest_data_dir, ratio, N)

def copy_test_yaml(conn, orig_data_dir, dest_data_dir, experiment_hash, ratio, N):
    test_matrix_ids = get_test_matrices_ids(conn, experiment_hash)
    for matrix_id in test_matrix_ids:
        matrix_file_name = os.path.join(orig_data_dir, str(matrix_id)+'.csv.gz')
        for i in range(N):
            dest_file_name = os.path.join(dest_data_dir, 'oversampled_frac_'+str(i), 'matrices', str(matrix_id)+'.csv.gz')
            print("Copying to:"+str(dest_file_name))
            shutil.copy(matrix_file_name, dest_file_name)

    print("Orig data dir="+str(os.path.join(orig_data_dir)))
    for in_file in os.listdir(os.path.join(orig_data_dir)):
        if(in_file.endswith("yaml")):
            for i in range(N):
                print("Copying:"+str(os.path.join(orig_data_dir,in_file))+" TO "+str(os.path.join(dest_data_dir, 'oversampled_frac_'+str(i))))
                shutil.copy(os.path.join(orig_data_dir, in_file), 
                os.path.join(dest_data_dir, 'oversampled_frac_'+str(i), 'matrices', in_file))

if __name__ == '__main__':
    DATA_DIR = '/mnt/data/experiment_data/peeps'
    conn = connect('../../config')
    orig_data_dir = os.path.join(DATA_DIR, 'joco_original', 'matrices')
    dest_data_dir = os.path.join(DATA_DIR, 'joco_oversampled')
    experiment_hash = sys.argv[0]
    ratio = 1
    N = int(sys.argv[2])
    mode = int(sys.argv[3])
    
    oversample_multiple_times(conn, orig_data_dir, dest_data_dir, experiment_hash, ratio, N, mode)
    copy_test_yaml(conn, orig_data_dir, dest_data_dir, experiment_hash, ratio, N)