import os, sys
import numpy as np
sys.path.insert(0, '../fair-classification/disparate_mistreatment/synthetic_data_demo')
sys.path.insert(0, '../fair-classification/fair_classification/')
from generate_synthetic_data import *
import utils as ut
import funcs_disp_mist as fdm
import plot_syn_boundaries as psb
import pandas as pd

FILE_PATH = '/mnt/data/experiment_data/peeps/joco_zafar/matrices'

label_col = 'booking_view_warr_bw_1y'
demo_col = 'race_2way'
exclude_cols = ['entity_id', 'as_of_date', label_col, demo_col]

def load_matrix(matrix_id, entity_to_attrib):
    df = pd.read_csv('%s/%s.csv.gz' % (FILE_PATH, matrix_id), compression='gzip')
    entity_col = []
    entities = df['entity_id'].values
    for i in range(len(entities)):
        attr = entities[i]
        try:
            entity_col.append(attr)
        except KeyError as e:
            entity_col.append("MISSING")

    df[demo_col] = entity_col
    df = df[df[demo_col]!='MISSING']
    df[label_col] = 2*(df[label_col] - 0.5)
    
    return df

def get_matrix_from_df(df, demo_col, label_col):
    exclude_cols = ['entity_id', 'as_of_date', label_col, demo_col]
    x = df[[c for c in df.columns if c not in exclude_cols]].values
    y = df[label_col].values
    x_control = {demo_col: df[demo_col].values}

    return x, y, x_control

def get_train_test_matrices(conn, experiment_hash):
    query = """
    select matrix_id, matrix_uuid, matrix_type from
    model_metadata.matrices where built_by_experiment='%s'
    """%(str(experiment_hash))

    df = pd.read_sql(query, conn)
    train_matrices = df[df['matrix_type'] == 'train']
    test_matrices = df[df['matrix_type'] == 'test']

    return train_matrices, test_matrices

def make_train_test_matrices(FILE_PATH, train_matrices, test_matrices):
    
    matrix_info = []
    
    for i in range(len(train_matrices)):
        train_matrix_end_time = train_matrices.iloc[i]['end_time']
        test_matrix_str = label_col+"_"+str(train_matrix_matrix_end_time)+"_"+str(train_matrix_end_time+relativedelta(years=1))

        if(os.path.exists(os.path.join(FILE_PATH, test_matrix_str))):
            print("Matrix Exists")            
            matrix_info.append([train_matrices.iloc[i]['matrix_uuid'], test_matrix_str])
        else:
            print("Some error")

    return matrix_info







