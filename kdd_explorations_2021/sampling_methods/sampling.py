import os
import sys
project_path = '../..'
sys.path.append(project_path)
import getpass
import ntpath
import logging
import time
from datetime import datetime
import yaml
import pandas as pd
import shutil
from sqlalchemy import create_engine
import numpy as np
from sampling_utils import *

def read_config_file(config_file):
    config = None
    try:
        with open (config_file, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(e)
        print('Error reading the config file')
    return config

def check_config_file(config):
    print(config['sampling_config'])

    experiment_hash = config['experiment_hash']
    assert experiment_hash is not None

    orig_dest_name = config['orig_dest_name']
    assert os.path.exists(orig_dest_name) == True

def connect(cred_folder):
    cred_file = os.path.join(cred_folder, 'san_jose_db.yaml')
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

def get_entities_attribute(conn, config):
    entity_to_attrib = {}
    query = """
    SELECT entity_id, %s 
    FROM %s
    """%(str(config['demo_col']), str(config['demo_table']))

    df = pd.read_sql(query, conn)
    entities = df['entity_id'].values
    attribs = df[config['demo_col']].values
    
    for i in range(len(entities)):
        entity_id = entities[i]
        attrib_info = attribs[i]
        entity_to_attrib[entity_id] = attrib_info
    
    return entity_to_attrib

def get_train_test_pairs_two(conn, experiment_ids, model_group_id):
    experiment_hashes = "','".join([str(x) for x in experiment_ids])
    experiment_hashes = "'"+experiment_hashes+"'"
    print(experiment_hashes)

    query = """
        with relevant_matrices as 
        (
            select experiment_hash, matrix_uuid from triage_metadata.experiment_matrices em where experiment_hash IN 
            (%s)
        )
        select experiment_hash,matrix_id, matrix_uuid, matrix_type from
        triage_metadata.matrices mm
        join relevant_matrices rm using (matrix_uuid);
    """%(str(experiment_hashes))

    df = pd.read_sql(query, conn)
    
    train_matrices = []
    test_matrices = []

    for exp_hash in experiment_ids:
        train_matrix_id = df[(df['experiment_hash'] == exp_hash) & (df['matrix_type']=='train')]['matrix_uuid'].values
        test_matrix_id = df[(df['experiment_hash'] == exp_hash) & (df['matrix_type']=='test')]['matrix_uuid'].values

        for m in train_matrix_id:
            train_matrices.append(m)
        for m in test_matrix_id:    
            test_matrices.append(m)

    return train_matrices, test_matrices    

def get_train_test_pairs(conn, experiment_id, model_group_id):
    query = """
    with rel_models as
    (
        select model_id, model_hash 
        from model_metadata.models
        where 
            built_by_experiment = '%s'
            and model_group_id = %s
    ),
    train_matrices as
    (
       select model_id, matrix_uuid from
       train_results_aug11.prediction_metadata
    ),
    test_matrices as 
    (
        select model_id, matrix_uuid from 
        test_results_aug11.prediction_metadata
    ),
    matrix_info as 
    (
        select matrix_id, matrix_uuid,
        matrix_type, num_observations
        from model_metadata.matrices
    )
    select 
        rel_models.model_id, 
        train_matrices.matrix_uuid as train_matrix_id,
        test_matrices.matrix_uuid as test_matrix_id,
        m1.matrix_id as train_id, 
        m1.num_observations as train_n_obs,
        m2.matrix_id as test_id,
        m2.num_observations as test_n_obs
    from 
        rel_models, 
        train_matrices, test_matrices,
        matrix_info m1, matrix_info m2
    where
        rel_models.model_id = train_matrices.model_id 
    and
        rel_models.model_id = test_matrices.model_id
    and 
        m1.matrix_uuid = train_matrices.matrix_uuid
    and 
        m2.matrix_uuid = test_matrices.matrix_uuid;
    """%(str(experiment_id), str(model_group_id))

    df = pd.read_sql(query, conn)

    vals = df.values
    train_matrices = []
    test_matrices = []

    for v in vals:
        train_matrices.append(v[1])
        test_matrices.append(v[2])
        #train_test_matrices.append([(v[1], v[2])])
        print(v[1], v[2])

    return train_matrices, test_matrices

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
    SELECT matrix_uuid from model_metadata.matrices where matrix_type='train' AND built_by_experiment = '%s'
    """%(str(experiment_id))

    df = pd.read_sql(query, conn)
    train_matrices = df['matrix_uuid'].values

    return train_matrices

def get_groups_from_train_matrix(matrix_id, orig_data_dir, entity_to_attrib, config):
    matrix_data = pd.read_csv(os.path.join(orig_data_dir+'/matrices', str(matrix_id)+'.csv.gz'), compression='gzip', error_bad_lines=False)
    print("MATRIX READ."+str(len(matrix_data)))

    entities = matrix_data['entity_id'].values
    as_of_dates = matrix_data['as_of_date'].values

    demo_col_arr = []
    missed = 0

    for i in range(len(entities)):
        entity_id = entities[i]
        try:
            demo_col_info = entity_to_attrib[entity_id]
            demo_col_arr.append(demo_col_info)
        except KeyError as e:
            missed = missed + 1
            demo_col_arr.append('MISSING')

    matrix_data[config['demo_col']] = demo_col_arr
    matrix_data = matrix_data[matrix_data[config['demo_col']]!='MISSING']
    
    attrib_1 = config['demo_permutations'][0]
    attrib_2 = config['demo_permutations'][1]

    dcol = config['demo_col']
    lcol = config['label']

    attrib_1_df_0 = matrix_data[(matrix_data[dcol]==attrib_1) & (matrix_data[lcol]==0.0)]
    attrib_1_df_1 = matrix_data[(matrix_data[dcol]==attrib_1) & (matrix_data[lcol]==1.0)]

    attrib_2_df_0 = matrix_data[(matrix_data[dcol]==attrib_2) & (matrix_data[lcol]==0)]
    attrib_2_df_1 = matrix_data[(matrix_data[dcol]==attrib_2) & (matrix_data[lcol]==1)]

    merged_df = {
        'a1':{
            'l0': attrib_1_df_0,
            'l1': attrib_1_df_1
        },
        'a2':{
            'l0': attrib_2_df_0,
            'l1': attrib_2_df_1
        }
    }

    entity_attrib_df = pd.DataFrame({
        'entity_id':matrix_data['entity_id'].values, 
        'attrib_values':matrix_data[dcol].values
    })
    
    return merged_df, entity_attrib_df, min(as_of_dates), max(as_of_dates)

def get_ideal_lengths(merged_df, sampling_config):
    print("A1 Stats")
    print(len(merged_df['a1']['l0']), len(merged_df['a1']['l1']), len(merged_df['a1']['l0'])+len(merged_df['a1']['l1']))
    print("A2 Stats")
    print(len(merged_df['a2']['l0']), len(merged_df['a2']['l1']), len(merged_df['a2']['l0'])+len(merged_df['a2']['l1']))

    n_a1 = len(merged_df['a1']['l0']) + len(merged_df['a1']['l1'])
    n_a2 = len(merged_df['a2']['l0']) + len(merged_df['a2']['l1'])
    
    if n_a1 < n_a2:
        protected = 'a1'
        non_protected = 'a2'
        n = {'protected': n_a1, 'non_protected': n_a2}
        n_p_l0 = float(len(merged_df[protected]['l0']))
        n_p_l1 = float(len(merged_df[protected]['l1']))
        n_np_l0 = float(len(merged_df[non_protected]['l0']))
        n_np_l1 = float(len(merged_df[non_protected]['l1']))
    else:
        protected = 'a2'
        non_protected = 'a1'
        n = {'protected': n_a2, 'non_protected': n_a1}
        n_p_l0 = float(len(merged_df[protected]['l0']))
        n_p_l1 = float(len(merged_df[protected]['l1']))
        n_np_l0 = float(len(merged_df[non_protected]['l0']))
        n_np_l1 = float(len(merged_df[non_protected]['l1']))

    mode = sampling_config['mode']
    orig_ratio = float(n['protected'])/float(n['non_protected'])
    orig_fraction = float(n['protected'])/(float(n['protected'] + n['non_protected']))
    
    ldist_p = n_p_l0/n_p_l1
    ldist_np = n_np_l0/n_np_l1
    ldist_p_ratio = float(n_p_l0)/(n_p_l0+n_p_l1)
    ldist_np_ratio = float(n_np_l0)/(n_np_l0+n_np_l1)

    req_ratio = sampling_config['ratio_protected']
    req_ldist_protected = sampling_config['label_dis_within_protected']
    req_ldist_nonprotected = sampling_config['label_dis_within_non_protected']

    if((req_ldist_protected == 'original') & (req_ldist_nonprotected == 'original')):
        
        if(req_ratio * n['protected'] < n['non_protected']):
            if(mode == 'under'):
                req_p = n['protected']
                req_np = req_ratio * req_p

            if(mode == 'over'):
                req_np = n['non_protected']
                req_p = int(req_np/float(req_ratio))

            req_p_l0 = int(float(ldist_p_ratio) * req_p)
            print("rpl0="+str(req_p_l0))
            req_p_l1 = int(req_p - req_p_l0)

            req_np_l0 = int(float(ldist_np_ratio) * req_np)
            req_np_l1 = int(req_np - 1 - req_np_l0)
            
            print(req_p, req_np, ldist_p_ratio, ldist_np_ratio, req_p_l0, req_p_l1, req_np_l0, req_np_l1)
            print("Required Stats")
            print("="*20)
            print(req_p_l0, req_p_l1, req_np_l0, req_np_l1)
            print("="*20)
            
            '''
            print("required stats")
            print(req_p_l0, req_p_l1, req_np_l0, req_np_l1)

            if(mode == 'under'):
                matrix_p_l0 = merged_df[protected]['l0'].sample(n=req_p_l0, replace=False)
                matrix_p_l1 = merged_df[protected]['l1'].sample(n=req_p_l1, replace=False)

                matrix_np_l0 = merged_df[non_protected]['l0'].sample(n=req_np_l0, replace=False)
                matrix_np_l1 = merged_df[non_protected]['l1'].sample(n=req_np_l1, replace=False)
            else:
                matrix_p_l0 = merged_df[protected]['l0'].sample(n=req_p_l0, replace=True)
                matrix_p_l1 = merged_df[protected]['l1'].sample(n=req_p_l1, replace=True)
                matrix_np_l0 = merged_df[non_protected]['l0'].sample(n=req_np_l0, replace=True)
                matrix_np_l1 = merged_df[non_protected]['l1'].sample(n=req_np_l1, replace=True)
            '''
        else:
            print("entering forbidden territory")
            req_p_l0 = int(n_p_l0)
            req_p_l1 = int(n_p_l1)
            req_np_l0 = int(n_np_l0)
            req_np_l1 = int(n_np_l1)
            #matrix_p_l0 = merged_df[protected]['l0']
            #matrix_p_l1 = merged_df[protected]['l1']
            #matrix_np_l0 = merged_df[non_protected]['l0']
            #matrix_np_l1 = merged_df[non_protected]['l1']

    if(req_ratio == "original"):
        req_frac = orig_fraction
        req_p = orig_fraction * n['protected']
        req_np = n['protected'] + n['non_protected'] - req_p

        ## VERSION 2A
        if(req_ldist_protected == '50' and req_ldist_nonprotected == '50'):
            if(mode=='under'):
                req_p_l0, req_p_l1, req_np_l0, req_np_l1 = sample_orig_50_50_undersample(orig_fraction, n_p_l0, n_p_l1, n_np_l0, n_np_l1)

            elif(mode=='over'):
                req_p_l0, req_p_l1, req_np_l0, req_np_l1 = sample_orig_50_50_oversample(orig_fraction, n_p_l0, n_p_l1, n_np_l0, n_np_l1)

        ## VERSION 2B
        if(req_ldist_protected == "50" and req_ldist_nonprotected == "original"):
            if(mode=='under'):
                req_p_l0, req_p_l1, req_np_l0, req_np_l1 = sample_orig_50_orig_undersample(orig_fraction, n_p_l0, n_p_l1, n_np_l0, n_np_l1)
            elif(mode=='over'):
                req_p_l0, req_p_l1, req_np_l0, req_np_l1 = sample_orig_50_orig_oversample(orig_fraction, n_p_l0, n_p_l1, n_np_l0, n_np_l1)
        
        ## VERSION 2C
        if(req_ldist_protected == 'same_nop'):
            if(mode == 'under'):
                req_p_l0, req_p_l1, req_np_l0, req_np_l1 = sample_orig_snop_orig_undersample(orig_fraction, n_p_l0, n_p_l1, n_np_l0, n_np_l1)
            elif(mode == 'over'):
                req_p_l0, req_p_l1, req_np_l0, req_np_l1 = sample_orig_snop_orig_oversample(orig_fraction, n_p_l0, n_p_l1, n_np_l0, n_np_l1)
    
    if(req_ratio == "1"):
        ## VERSION 3A
        if(req_ldist_protected == "50" and req_ldist_nonprotected == "50"):
            if(mode == "under"):
                req_p_l0, req_p_l1, req_np_l0, req_np_l1 = sample_1_50_50_undersample(orig_fraction, n_p_l0, n_p_l1, n_np_l0, n_np_l1)
            elif(mode == "over"):
                req_p_l0, req_p_l1, req_np_l0, req_np_l1 = sample_1_50_50_oversample(orig_fraction, n_p_l0, n_p_l1, n_np_l0, n_np_l1)
        
        ## VERSION 3B
        if(req_ldist_protected == "same_nop" and req_ldist_nonprotected == "original"):
            if(mode == "under"):
                req_p_l0, req_p_l1, req_np_l0, req_np_l1 = sample_1_snop_orig_undersample(orig_fraction, n_p_l0, n_p_l1, n_np_l0, n_np_l1)
            elif(mode == "over"):
                req_p_l0, req_p_l1, req_np_l0, req_np_l1 = sample_1_snop_orig_oversample(orig_fraction, n_p_l0, n_p_l1, n_np_l0, n_np_l1)

    print("Sampling")
    print(req_p_l0, req_p_l1, req_np_l0, req_np_l1)
    if(n_p_l0 < req_p_l0):
        matrix_p_l0 = merged_df[protected]['l0'].sample(n=req_p_l0, replace=True)
    else:
        matrix_p_l0 = merged_df[protected]['l0'].sample(n=req_p_l0, replace=False)

    if(n_p_l1 < req_p_l1):
        matrix_p_l1 = merged_df[protected]['l1'].sample(n=req_p_l1, replace=True)
    else:
        matrix_p_l1 = merged_df[protected]['l1'].sample(n=req_p_l1, replace=False)

    if(n_np_l0 < req_np_l0):
        matrix_np_l0 = merged_df[non_protected]['l0'].sample(n=req_np_l0, replace=True)
    else:
        matrix_np_l0 = merged_df[non_protected]['l0'].sample(n=req_np_l0, replace=False)

    if(n_np_l1 < req_np_l1):
        matrix_np_l1 = merged_df[non_protected]['l1'].sample(n=req_np_l1, replace=True)
    else:
        matrix_np_l1 = merged_df[non_protected]['l1'].sample(n=req_np_l1, replace=False)
    
    return matrix_p_l0, matrix_p_l1, matrix_np_l0, matrix_np_l1, [n_p_l0, n_p_l1, n_np_l0, n_np_l1]

def set_entity_ids(overall_df, multiplier):
    mod_entity_ids = []
    entity_ids = overall_df['entity_id'].values
    for i in range(len(entity_ids)):
        mod_entity_ids.append((entity_ids[i] * multiplier) + (i+1))

    overall_df['entity_id'] = mod_entity_ids
    return overall_df

def sample_for_all_matrices(conn, orig_data_dir, entity_to_attrib, config, SAMPLING_DETAILS_DF):
    sampling_config = config['sampling_config']

    train_matrix_ids, test_matrix_ids = get_train_test_pairs_two(conn, config['experiment_hash'], config['model_group_id'])

    print(len(train_matrix_ids), len(test_matrix_ids))
    dest_data_dir = os.path.join("/mnt/data/experiment_data/san_jose_housing",

    sampling_config['mode']+"_"+str(sampling_config['ratio_protected'])+"_"+str(sampling_config['label_dis_within_protected'])+"_"+str(sampling_config['label_dis_within_non_protected']), 'matrices')

    print("Output TO:"+str(dest_data_dir))
    if(not os.path.exists(dest_data_dir)):
        os.makedirs(dest_data_dir)        
    
    for i in range(len(train_matrix_ids)):
        print(str(i)+"/"+str(len(train_matrix_ids)))
        
        merged_df, entity_attrib_df, min_date, max_date = get_groups_from_train_matrix(train_matrix_ids[i], orig_data_dir, entity_to_attrib, config)

        matrix_p_l0, matrix_p_l1, matrix_np_l0, matrix_np_l1, orig_dims = get_ideal_lengths(merged_df, config['sampling_config'])

        SAMPLING_DETAILS_DF.append([sampling_config['mode'], sampling_config['ratio_protected'], sampling_config['label_dis_within_protected'], sampling_config['label_dis_within_non_protected'], min_date, max_date, train_matrix_ids[i], test_matrix_ids[i], orig_dims[0], orig_dims[1], orig_dims[2], orig_dims[3], len(matrix_p_l0), len(matrix_p_l1), len(matrix_np_l0), len(matrix_np_l1)])

        #write_details_to_file(fw, train_matrix_ids[i], test_matrix_ids[i], matrix_p_l0, matrix_p_l1, matrix_np_l0, matrix_np_l1, orig_dims)

        overall_df = pd.concat([matrix_p_l0, matrix_p_l1, matrix_np_l0, matrix_np_l1])

        overall_df = set_entity_ids(overall_df, config['multiplier'])
        filtered_df = overall_df.drop(config['demo_col'], 1)

        filtered_df.to_csv(os.path.join(dest_data_dir, str(train_matrix_ids[i])+".csv.gz"), compression='gzip', index=False)

        train_matrix_yaml = os.path.join(orig_data_dir, 'matrices', str(train_matrix_ids[i])+
        ".yaml")
        dest_matrix_yaml = os.path.join(dest_data_dir, str(train_matrix_ids[i])+".yaml")
        
        shutil.copy(train_matrix_yaml, dest_matrix_yaml)

    for i in range(len(test_matrix_ids)):
        test_matrix_file = os.path.join(orig_data_dir, 'matrices', str(test_matrix_ids[i])+'.csv.gz')
        dest_matrix_file = os.path.join(dest_data_dir, str(test_matrix_ids[i])+'.csv.gz')
        
        test_yaml_file = os.path.join(orig_data_dir, 'matrices', str(test_matrix_ids[i])+'.yaml')
        dest_yaml_file = os.path.join(dest_data_dir, str(test_matrix_ids[i])+'.yaml')

        shutil.copy(test_matrix_file, dest_matrix_file)
        shutil.copy(test_yaml_file, dest_yaml_file)

    return SAMPLING_DETAILS_DF

def all_configs():
    configs = {}
       
    configs['v1a_u'] = {'mode':'under', 'ratio_protected':1, 'label_dis_within_protected': 'original', 'label_dis_within_non_protected':'original'}
    
    #configs['v1b_u'] = {'mode':'under', 'ratio_protected':2, 'label_dis_within_protected': 'original', 'label_dis_within_non_protected':'original'}

    #configs['v1c_u'] = {'mode':'under', 'ratio_protected':3, 'label_dis_within_protected': 'original', 'label_dis_within_non_protected':'original'}
    
    configs['v1a_o'] = {'mode':'over', 'ratio_protected':1, 'label_dis_within_protected': 'original', 'label_dis_within_non_protected':'original'}

    #configs['v1b_o'] = {'mode':'over', 'ratio_protected':2, 'label_dis_within_protected': 'original', 'label_dis_within_non_protected':'original'}

    #configs['v1c_o'] = {'mode':'over', 'ratio_protected':3, 'label_dis_within_protected': 'original', 'label_dis_within_non_protected':'original'}

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
    
    return configs

if __name__ == '__main__':
    conn = connect('../../config')

    config_file = sys.argv[1]
    config = read_config_file(config_file)
    
    base_dir = config['base_dir']
    orig_data_dir = os.path.join(base_dir, config['orig_data_name'])

    #sampling_config = config['sampling_config']
    
    entity_to_attrib = get_entities_attribute(conn, config)
    
    configs = all_configs()

    SAMPLING_DETAILS_DF = []
    
    for key in configs.keys():
        config['sampling_config'] = configs[key]
        SAMPLING_DETAILS_DF = sample_for_all_matrices(conn, orig_data_dir, entity_to_attrib, config, SAMPLING_DETAILS_DF)

    SAMPLING_DETAILS_DF = pd.DataFrame(SAMPLING_DETAILS_DF, columns = ['mode', 'ratio_protected', 'ldist_p', 'ldist_np', 'min_as_of_date', 'max_as_of_date', 'train_matrix_id', 'test_matrix_id', 'orig_p_l0', 'orig_p_l1', 'orig_np_l0', 'orig_np_l1', 'sampled_p_l0', 'sampled_p_l1', 'sampled_np_l0', 'sampled_np_l1'])

    SAMPLING_DETAILS_DF.to_csv(os.path.join(base_dir, 'SAMPLING_DETAILS_DATAFRAME.csv'))
