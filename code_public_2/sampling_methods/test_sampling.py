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
from sampling import get_ideal_lengths, sample_orig_snop_orig_Undersample, sample_orig_snop_orig_Oversample

'''
sampling_config:
ratio_protected: 1
label_dis_within_protected: 'original'
label_dis_within_non_protected: 'original'
mode: 'under'
'''

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

def read_config_file(config_file):
    config = None
    try:
        with open (config_file, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(e)
        print('Error reading the config file')
    return config

def for_all_sampling_configs():
    configs = {}
    
    configs['v1a_u'] = {'mode':'under', 'ratio_protected':1, 'label_dis_within_protected': 'original', 'label_dis_within_non_protected':'original'}

    configs['v1b_u'] = {'mode':'under', 'ratio_protected':2, 'label_dis_within_protected': 'original', 'label_dis_within_non_protected':'original'}

    configs['v1c_u'] = {'mode':'under', 'ratio_protected':3, 'label_dis_within_protected': 'original', 'label_dis_within_non_protected':'original'}

    configs['v1a_o'] = {'mode':'over', 'ratio_protected':1, 'label_dis_within_protected': 'original', 'label_dis_within_non_protected':'original'}

    configs['v1b_o'] = {'mode':'over', 'ratio_protected':2, 'label_dis_within_protected': 'original', 'label_dis_within_non_protected':'original'}

    configs['v1c_o'] = {'mode':'over', 'ratio_protected':3, 'label_dis_within_protected': 'original', 'label_dis_within_non_protected':'original'}

    configs['v2a_u'] = {'mode':'under', 'ratio_protected':'original', 'label_dis_within_protected': '50', 'label_dis_within_non_protected':'50'}
    
    configs['v2a_o'] = {'mode':'over', 'ratio_protected':'original', 'label_dis_within_protected': '50', 'label_dis_within_non_protected':'50'}
    
    configs['v2b_u'] = {'mode':'under', 'ratio_protected':'original', 'label_dis_within_protected': 'same_nop', 'label_dis_within_non_protected':'original'}
    
    configs['v2b_o'] = {'mode':'over', 'ratio_protected':'original', 'label_dis_within_protected': 'same_nop', 'label_dis_within_non_protected':'original'}

    configs['v2c_u'] = {'mode':'under', 'ratio_protected':'original', 'label_dis_within_protected': '50', 'label_dis_within_non_protected':'original'}
    
    configs['v2c_o'] = {'mode':'over', 'ratio_protected':'original', 'label_dis_within_protected': '50', 'label_dis_within_non_protected':'original'}

    configs['v3a_u'] = {'mode':'under', 'ratio_protected':'1', 'label_dis_within_protected': '50', 'label_dis_within_non_protected':'50'}
    
    configs['v3a_o'] = {'mode':'over', 'ratio_protected':'1', 'label_dis_within_protected': '50', 'label_dis_within_non_protected':'50'}

    configs['v3b_u'] = {'mode':'under', 'ratio_protected':'1', 'label_dis_within_protected': 'same_nop', 'label_dis_within_non_protected':'original'}
    
    configs['v3b_o'] = {'mode':'over', 'ratio_protected':'1', 'label_dis_within_protected': 'same_nop', 'label_dis_within_non_protected':'original'}

    for key in configs.keys():
        print("="*30)
        print(configs[key])
        print("="*30)
        test(configs[key])

def test(sampling_config):

    merged_df = {
        'a1':{
            'l0': pd.DataFrame({"data":[5.0] * 200}),
            'l1': pd.DataFrame({"data":[1.0] * 800})
        },
        'a2':{
            'l0': pd.DataFrame({"data":[10.0] * 400}),
            'l1': pd.DataFrame({"data":[20.0] * 3200})
        }
    }

#    merged_df = {
#        'a1':{
#            'l0': pd.DataFrame({"data":[5.0] * 200}),
#            'l1': pd.DataFrame({"data":[1.0] * 800})
#        },
#       'a2':{
#           'l0': pd.DataFrame({"data":[10.0] * 200}),
#           'l1': pd.DataFrame({"data":[20.0] * 1000})
#       }
#    }

    matrix_p_l0, matrix_p_l1, matrix_np_l0, matrix_np_l1 = get_ideal_lengths(merged_df, sampling_config)

    overall_df = pd.concat([matrix_p_l0, matrix_p_l1, matrix_np_l0, matrix_np_l1])

    overall_df['entity_id'] = range(len(overall_df))
    return overall_df

def get_train_matrices_ids(conn, experiment_id):
    query = """
    SELECT matrix_uuid from model_metadata.matrices where matrix_type='train' AND
    built_by_experiment = '%s'
    """%(str(experiment_id))

    df = pd.read_sql(query, conn)
    train_matrices = df['matrix_uuid'].values

    return train_matrices

def get_groups_from_train_matrix(matrix_file, config):
    matrix_data = pd.read_csv(matrix_file, compression='gzip', error_bad_lines=False)
    
    entities = matrix_data['entity_id'].values
    attributes = matrix_data[config['orig_demo_col']].values


    demo_col_arr = []
    missed = 0
    for i in range(len(entities)):
        entity_id = entities[i]
        attribute_value = attributes[i]

        if(attribute_value == 1):
            demo_col_arr.append('White')
        else:
            demo_col_arr.append('NonWhite')
    
    matrix_data[config['demo_col']] = demo_col_arr
    matrix_data = matrix_data[matrix_data[config['demo_col']]!='MISSING']

    attrib_1 = config['demo_permutations'][0]
    attrib_2 = config['demo_permutations'][1]

    dcol = config['demo_col']
    lcol = config['label']

    attrib_1_df_0 = matrix_data[(matrix_data[dcol]==attrib_1) & (matrix_data[lcol]==0)]
    attrib_1_df_1 = matrix_data[(matrix_data[dcol]==attrib_1) & (matrix_data[lcol]==1)]

    attrib_2_df_0 = matrix_data[(matrix_data[dcol]==attrib_2) & (matrix_data[lcol]==0)]
    attrib_2_df_1 = matrix_data[(matrix_data[dcol]==attrib_2) & (matrix_data[lcol]==1)]

    return attrib_1_df_0, attrib_1_df_1, attrib_2_df_0, attrib_2_df_1

def compute_metrics(df, label_col, demo_col):
    entity_ids = df['entity_id'].values
    print(pd.value_counts(df[label_col]))
    print(pd.value_counts(df[demo_col]))

def display_results_generated_matrix(base_dir, test_matrices, config, dest_fname):
    fw = open(dest_fname, 'w')
    ds_names = os.listdir(base_dir)
    for ds_name in ds_names:
        print(ds_name)
        if((not ds_name.startswith("joco")) and (ds_name!='SAMPLING_DETAILS')):
            fw.write(str(ds_name)+"\n")
            fw.write("-"*30+'\n')
            for i in range(len(test_matrices)):
                print("\tMatrix ID="+str(test_matrices[i]))
                matrix_file = os.path.join(base_dir, ds_name, 'matrices', test_matrices[i]+'.csv.gz')

                df_a1_l0, df_a1_l1, df_a2_l0, df_a2_l1 = get_groups_from_train_matrix(matrix_file, config)
                df = pd.read_csv(matrix_file, compression='gzip')
                print(len(df_a1_l0), len(df_a1_l1), len(df_a2_l0), len(df_a2_l1))
                fw.write("Matrix ID:"+str(test_matrices[i])+'\n')
                fw.write(str(len(df_a1_l0))+' , '+str(len(df_a1_l1))+'\t'+str(len(df_a2_l0))+' , '+str(len(df_a2_l1))+'\n')
                fw.flush()
    fw.close()  

if __name__ == '__main__':
    conn = connect('../../config')
    config_file = sys.argv[1]
    config = read_config_file(config_file)
    
    test_matrices = get_train_matrices_ids(conn, config['experiment_hash'])
    label_col = config['label']
    demo_col = config['demo_col']
    
    display_results_generated_matrix(config['base_dir'], test_matrices, config, '/mnt/data/experiment_data/joco_sampling.txt')
