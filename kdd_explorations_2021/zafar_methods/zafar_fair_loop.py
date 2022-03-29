import os, sys
import json
import datetime
import numpy as np
sys.path.insert(0, '../fair-classification/disparate_mistreatment/synthetic_data_demo')
sys.path.insert(0, '../fair-classification/fair_classification/')
from generate_synthetic_data import *
import utils as ut
import funcs_disp_mist as fdm
import plot_syn_boundaries as psb
import pandas as pd
import cvxpy
import yaml
from triage import create_engine
from triage.component.catwalk.estimators.classifiers import ScaledLogisticRegression
import time
from psycopg2.extras import Json


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
    #cred_file = os.path.join(cred_folder, 'donors_db_profile.yaml')
    #cred_file = os.path.join(cred_folder, 'elsal_db_profile2.yaml')
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

def load_matrix(FILE_PATH, matrix_id, entity_to_attrib, demo_col, label_col):
    print("Reading:"+str(FILE_PATH)+"/"+str(matrix_id)+".csv.gz")

    df = pd.read_csv('%s/%s.csv.gz' % (FILE_PATH, matrix_id), compression='gzip')

    entity_col = []
    entities = df['entity_id'].values
    
    for i in range(len(entities)):
        try:
            attr = entity_to_attrib[int(entities[i])]
            entity_col.append(attr)
        except KeyError as e:
            entity_col.append("MISSING")

    df[demo_col] = entity_col
    df = df[df[demo_col]!='MISSING']
    df[label_col] = 2*(df[label_col] - 0.5)

    return df

def get_matrix_from_df(df, demo_col, label_col, cols_0):
    exclude_cols = ['entity_id', 'as_of_date', label_col, demo_col]
    
    for c in df.columns:
        if c in cols_0:
            exclude_cols.append(c)

    x = df[[c for c in df.columns if c not in exclude_cols]].values
    y = df[label_col].values
    x_control = {demo_col:df[demo_col].values}

    return x, y, x_control

def myconverter(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()

def get_train_test_matrix_pairs(engine, experiment_hash, model_group_id):
    '''
    args:
        engine: PSQL Connection Engine
        experiment_hash: use the relevant experiment_hash attributing to the train/test matrices you want to obtain
        model_group_id: it might be better to use a model_group_id corresponding to 
        the dummy classifier for which train/test matrices might have been created; otherwise any relevant model_group_id
        will do.
    '''

    query = """
    with rel_models as
    (
        select model_id, model_hash 
        from triage_metadata.models
        where 
            built_by_experiment = '%s'
            and model_group_id = %s
    ),
    train_matrices as
    (
       select model_id, matrix_uuid from
       train_results.prediction_metadata
    ),
    test_matrices as 
    (
        select model_id, matrix_uuid from 
        test_results.prediction_metadata
    ),
    matrix_info as 
    (
        select matrix_id, matrix_uuid,
        matrix_type, num_observations
        from triage_metadata.matrices
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
    """%(str(experiment_hash), str(model_group_id))

    print(query)
    df = pd.read_sql(query, engine)
    print(df.head(2))
    
    vals = df.values
    train_test_matrices = []
    
    for v in vals:
        train_test_matrices.append([(v[1], v[2])])

    return train_test_matrices

def get_entity_to_attrib_simple(conn, race_query, sa_column, anchor_sa_value):
    entity_to_attrib = {}
    df = pd.read_sql(race_query, conn)
    
    entity_ids = df['entity_id'].values
    sa_vals = df[sa_column].values
    
    attrib_info = {}

    for i in range(len(entity_ids)):
        if(sa_vals[i]==anchor_sa_value):
            sa_vals[i] = 1
        else:
            sa_vals[i] = 0.0
        attrib_info[int(entity_ids[i])] = sa_vals[i]
        
    print(pd.value_counts(sa_vals))
    return attrib_info

def save_output_info(out_path, test_matrix_uuid, train_score, test_score,cov_all_train, cov_all_test, s_attr_to_fp_fn_train, s_attr_to_fp_fn_test, recall_highest, recall_nonhighest):

    out_file = open(os.path.join(out_path, test_matrix_uuid+"_Scores"), 'w')
    print("SAVING TO:"+str(os.path.join(out_path, test_matrix_uuid+"_Scores")))

    out_file.write(str(train_score)+"\t"+str(test_score)+"\n")
    out_file.write(str(cov_all_train)+'\n')
    out_file.write(str(cov_all_test)+'\n')
    out_file.write(str(s_attr_to_fp_fn_train)+'\n')
    out_file.write(str(s_attr_to_fp_fn_test)+'\n')
    out_file.flush()
    out_file.close()

    with open(os.path.join(out_path, "INFO_"+str(test_matrix_uuid)), 'w') as fp:
        json.dump({
            'train_score':train_score,
            'test_score':test_score,
            'cov_all_train':cov_all_train,
            'cov_all_test':cov_all_test,
            's_attr_to_fp_fn_train':s_attr_to_fp_fn_train,
            's_attr_to_fp_fn_test':s_attr_to_fp_fn_test,
            'recall_highest':recall_highest,
            'recall_nonhighest':recall_nonhighest
        }, fp, default=myconverter)

def label_top_k(distances, k):
    sort_df = pd.DataFrame({'dist': distances})
    sort_df.sort_values('dist', ascending=False, inplace=True)
    sort_df['pred_label'] = -1
    sort_df['orig_idx'] = sort_df.index
    sort_df.reset_index(inplace=True)
    i = k-1
    sort_df.loc[:i,'pred_label'] = 1
    sort_df.sort_values('orig_idx', inplace=True)
    
    return sort_df['pred_label'].values

def calc_prec(pred_label, actual_label):
    label_pos = sum((pred_label == 1).astype(int))
    true_pos = sum(np.logical_and(pred_label == actual_label, pred_label == 1).astype(int))
    return float(true_pos/label_pos)

def run_for_all_train_test_matrices(file_path, train_config,train_test_matrices, entity_to_attrib, out_path, test_results_table, test_results_schema):

    print("TRUNCATING........")
    truncate_query = """
    truncate table triage_metadata.zafar_models;
    """
    conn.execute(truncate_query)

    truncate_query = """
    truncate table triage_metadata.zafar_model_groups;
    """
    conn.execute(truncate_query)

    truncate_query = """
    truncate table %s.%s
    """%(str(test_results_schema),str(test_results_table))
    conn.execute(truncate_query)

    model_group_id = 9999
    df_model_group_insert = pd.DataFrame({
        'model_group_id': [model_group_id],
        'model_type': 'zafar_model_group',
        'hyperparameters': [Json(train_config)]
        #'feature_list': [],
        #'model_config: []
    })

    df_model_group_insert.to_sql('zafar_model_groups', conn, schema='triage_metadata', index=False, if_exists='append')
    
    for i in range(len(train_test_matrices)):
        print("Doing for "+str(i))
        print("--"*20)
        train_matrix_uuid = train_test_matrices[i][0][0]
        test_matrix_uuid = train_test_matrices[i][0][1]

        out_file = os.path.join(out_path, test_matrix_uuid+"_Scores")
        if (not os.path.exists(out_file)):
            df_train = load_matrix(file_path, train_matrix_uuid, entity_to_attrib, demo_col, label_col)
            
            # converting as of date
            df_train['as_of_date'] = pd.to_datetime(df_train['as_of_date'])
            
            # getting entities and as of dates
            train_as_of_dates = df_train['as_of_date'].values
            train_entity_ids = df_train['entity_id'].values

            # doing the same for df_test
            df_test = load_matrix(file_path, test_matrix_uuid, entity_to_attrib, demo_col, label_col)

            df_test['as_of_date'] = pd.to_datetime(df_test['as_of_date'])
            
            test_as_of_dates = df_test['as_of_date'].values
            test_entity_ids = df_test['entity_id'].values

            # Adding intercept
            df_train['intercept'] = 1
            df_test['intercept'] = 1
            
            # This filtering of columns is different than for what we will run zafar.
            old_exclude_cols = ['entity_id', 'as_of_date', label_col]

            # This is the matrix for which we will run Regression
            x_temp = df_train[[c for c in df_train.columns if c not in old_exclude_cols]].values
            y_temp = df_train[label_col].values
            
            # Perform Scaled Logistic Regression to include only relevant features.
            dsapp_lr = ScaledLogisticRegression(penalty = "l1", C=0.1)
            dsapp_lr.fit(x_temp, y_temp)

            all_columns = [c for c in df_train.columns if c not in old_exclude_cols]
            keep_cols = []

            for i, col in enumerate(all_columns):
                if dsapp_lr.coef_[0][i]!=0:
                    keep_cols.append(col)
        
            keep_cols = keep_cols + ['intercept']

            x_train = df_train[[c for c in df_train.columns if c in keep_cols]].values
            y_train = df_train[label_col].values
            x_control_train = {demo_col: df_train[demo_col].values}
            
            x_test = df_test[[c for c in df_test.columns if c in keep_cols]].values
            y_test = df_test[label_col].values
            x_control_test = {demo_col: df_test[demo_col].values}

            x = x_train
            y = y_train
            x_control = x_control_train

            max_iters = train_config['max_iters']
            max_iters_dccp = train_config['max_iters_dccp']
        
            num_points, num_features = x.shape
            w = cvxpy.Variable(num_features)

            np.random.seed(train_config['random_seed'])
            w.value = np.random.rand(x.shape[1])

            constraints = []
            loss = cvxpy.sum(cvxpy.logistic( cvxpy.multiply(-y, x*w) )  ) / num_points
            prob = cvxpy.Problem(cvxpy.Minimize(loss), constraints)

            tau =  float(train_config['tau'])
            mu = float(train_config['mu'])

            loss_function = "logreg" # perform the experiments with logistic regression
            #EPS = 1e-4
            EPS = float(train_config['EPS'])

            prob.solve(method='dccp', tau=tau, mu=mu, tau_max=1e10, solver=cvxpy.ECOS, verbose=True, feastol=EPS, abstol=EPS, reltol=EPS, feastol_inacc=EPS, abstol_inacc=EPS, reltol_inacc=EPS,max_iters=max_iters, max_iter=max_iters_dccp)

            ret_w = np.array(w.value).flatten()
            sensitive_attrs = list(x_control_train.keys())

            print("INPUT TRAIN:"+str(pd.value_counts(y_train)))
            print("INPUT TEST:"+str(pd.value_counts(y_test)))

            train_score, test_score, cov_all_train, cov_all_test, s_attr_to_fp_fn_train, s_attr_to_fp_fn_test = fdm.get_clf_stats(ret_w, x_train, y_train, x_control, x_test, y_test, x_control_test, list(sensitive_attrs))

            s_attr = sensitive_attrs[0]
            distances_boundary_test = fdm.get_distance_boundary(ret_w, x_test, x_control_test[s_attr])

            print("SAVING PREDICTIONS")
            save_predictions(distances_boundary_test, y_test, x_control_test, sensitive_attrs, test_as_of_dates, test_entity_ids, train_matrix_uuid, test_matrix_uuid, train_config, test_results_table, test_results_schema, out_path)
        
            k = 500
            all_class_labels_assigned_test = label_top_k(distances_boundary_test, k)
            #all_class_labels_assigned_test = np.sign(distances_boundary_test)

            prec_k = calc_prec(all_class_labels_assigned_test, y_test)
            print('prec@%s_abs: %.5f' % (k, prec_k))

            s_attr_to_fp_fn_test = fdm.get_fpr_fnr_sensitive_features(y_test, all_class_labels_assigned_test, x_control_test, sensitive_attrs, False)
        
            for s_attr in s_attr_to_fp_fn_test.keys():
                print("S_Attr="+str(s_attr))
                for s_val in s_attr_to_fp_fn_test[s_attr].keys():
                    print("S_VAL="+str(s_val))
                    s_attr_to_fp_fn_test[s_attr][s_val]['recall'] = 1.000-s_attr_to_fp_fn_test[s_attr][s_val]['fnr']

                #recall_white = s_attr_to_fp_fn_test['race'][0]['recall']
                #recall_nonwhite = s_attr_to_fp_fn_test['race'][1]['recall']
                #recall_highest = s_attr_to_fp_fn_test['plevel'][0]['recall']
                #recall_nonhighest = s_attr_to_fp_fn_test['plevel'][1]['recall']
                #recall_overage = s_attr_to_fp_fn_test['ovg'][0]['recall']
                #recall_non_overage = s_attr_to_fp_fn_test['ovg'][1]['recall']
                recall_under = s_attr_to_fp_fn_test['median_income'][0]['recall']
                recall_over = s_attr_to_fp_fn_test['median_income'][1]['recall']

                #print('recall white: %.6f' % recall_white)
                #print('recall non-white: %.6f' % recall_nonwhite)
                #print('recall ratio: %.6f' % float(recall_white/recall_nonwhite))
                #print('recall highest: %.6f' % recall_highest)
                #print('recall non-highest: %.6f' % recall_nonhighest)
                #print('recall ratio: %.6f' % float(recall_highest/(recall_nonhighest+1e-6)))
                #print('recall overage: %.6f' % recall_overage)
                #print('recall non overage: %.6f' % recall_non_overage)
                print('recall under55k: %.6f' % recall_under)
                print('recall over55k: %.6f' % recall_over)
        
            save_output_info(out_path, test_matrix_uuid, train_score, test_score, cov_all_train, cov_all_test, s_attr_to_fp_fn_train, s_attr_to_fp_fn_test, recall_under, recall_over)

    return None

def save_predictions(distances, y_test, x_control_test, sensitive_attrs, test_as_of_dates, test_entity_ids, train_matrix_uuid, test_matrix_uuid, 
train_config, test_results_table, test_results_schema, out_path):

    model_id_query = pd.read_sql("""
    select max(model_id) from triage_metadata.zafar_models
    """, conn)
    max_model_id = model_id_query['max'].values[0]

    if(max_model_id == None):
        model_id = 0
    else:
        model_id = max_model_id + 1

    print("SAVING FOR MODEL_ID="+str(model_id))

    model_group_id = 9999   #using this value as a proxy for model_group_id
    
    df_model_insert = pd.DataFrame({
        'model_id': model_id,
        'model_group_id':[model_group_id],
        'run_time':[datetime.datetime.now()],
        'model_type': ['zafar_regularized'],
        'hyperparameters': [Json(train_config)],
        'train_end_time': [max(test_as_of_dates)],
        'train_matrix_uuid': [train_matrix_uuid],
        'test_matrix_uuid': [test_matrix_uuid]
    })

    df_model_insert.to_sql('zafar_models', conn, schema='triage_metadata', index = False, if_exists='append')

    for s in sensitive_attrs:
        s_attr_vals = x_control_test[s]        
        
        df = pd.DataFrame({
            'model_id':[model_id] * len(test_entity_ids),
            'matrix_id':[test_matrix_uuid]*len(test_entity_ids), 'entity_id':test_entity_ids, 'as_of_date':test_as_of_dates, 'dist':np.array(distances), 'label': (1.0 + y_test)/2, 'x_control': s_attr_vals}
        )
        
        df.to_csv(os.path.join(out_path,'test_dataframe'+'_'+str(s)+'_'+str(test_matrix_uuid)+'.csv'))

        df.to_sql(test_results_table, conn, schema=test_results_schema, index=False, if_exists ='append')

        print(min(df['model_id'].values), max(df['model_id'].values))

if __name__ == "__main__":
    conn = connect('../../config')
    
    config_file = sys.argv[1]
    config = read_config_file(config_file)

    FILE_PATH = config['file_path']
    SAVE_FILE_PATH = config['out_file_path']
    label_col = config['label_col']
    demo_col = config['demo_col']
    exp_hash = config['experiment_hash']
    model_group_id = config['model_group_id']

    test_results_table = config['dest_table']
    test_results_schema = config['dest_schema']

    race_query = config['race_query']
    '''select entity_id, race
    from hemank_bias_alternatives.currmatch_entity_demos
    '''
    # sa_column = 'race'
    # sa_val = 'White'
    sa_column = config['sa_column']
    sa_value = config['anchor_sa_value']

    entity_to_attrib = get_entity_to_attrib_simple(conn, race_query, sa_column, sa_value)

    train_test_matrices = get_train_test_matrix_pairs(conn, exp_hash, model_group_id)

    print(train_test_matrices)
    out_path = config['out_file_path']
    
    run_for_all_train_test_matrices(FILE_PATH, config['train_config'], train_test_matrices, entity_to_attrib, out_path, test_results_table, test_results_schema)