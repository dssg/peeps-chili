import os
import sys
import getpass
import ntpath
import logging
import time
import yaml
from datetime import datetime
from triage.experiments import SingleThreadedExperiment
from triage.experiments import MultiCoreExperiment
from triage.component.timechop.timechop import Timechop
from triage.component.timechop.plotting import visualize_chops
from triage import create_engine

logging.basicConfig(level=logging.DEBUG, filename="LOG_Triage.debug", filemode='w')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def read_config_file(config_file):
    config = None
    try:
        with open (config_file, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(e)
        print('Error reading the config file')

    return config


def setup_experiment(experiment_config_file, use_s3, s3_path):
    project_path = ''
    experiment_config = read_config_file(experiment_config_file)

    cred_folder = os.path.join(project_path, 'conf', 'local')
    cred_file = os.path.join(cred_folder, 'credentials_2.yaml')
    
    configs = read_config_file(cred_file)
    db = configs['db']

    sql_engine = create_engine(
        'postgresql+psycopg2://%s:%s@%s:%i/%s'%(
            db['user'],
            db['pass'],
            db['host'],
            db['port'],
            db['db']
        )
    )

    dateTimeObj = datetime.now()
    timestr = dateTimeObj.strftime('%Y%m%d%H%M')
    user = getpass.getuser()
    cf = ntpath.basename(experiment_config_file)[0:10]
    
    if use_s3 == False:
        data_folder = '/mnt/data/experiment_data'
        project_folder = os.path.join(data_folder, 'triage', 'elsal') #'{}_{}_{}'.format(user, timestr, cf))

        if not os.path.exists(project_folder):
            os.mkdir(project_folder)
    else:
        project_folder = s3_path
    print(project_folder)
    return experiment_config, sql_engine, project_folder

def run_exp(config_file, plot_timechops=True, run_exp=True, use_s3=False, s3_path=None, n_jobs=1):
    if plot_timechops:
        visualize_timechop(config_file)

    config, sql_engine, proj_folder = setup_experiment(config_file, use_s3, s3_path)
    print("Project Folder="+str(proj_folder))
    #print(STOP)

    if run_exp: 
        if n_jobs> 1:
            experiment = MultiCoreExperiment(
                config=config,
                db_engine=sql_engine,
                n_processes=n_jobs,
                n_db_processes=n_jobs,
                project_path=proj_folder,
                replace=False
            )
        else:
            experiment = SingleThreadedExperiment(
                config=config,
                db_engine=sql_engine,
                project_path=proj_folder,
                cleanup=True,
                replace=False
            )

        st = time.time()
        experiment.run()
        en = time.time()

        print('Took {} seconds to run the experiement'.format(en-st))

def visualize_timechop(config_file):
    experiment_config = read_config_file(config_file)
    temporal_config = experiment_config['temporal_config']
    feature_start_time = temporal_config['feature_start_time']
    feature_end_time = temporal_config['feature_end_time']
    label_start_time = temporal_config['label_start_time']
    label_end_time = temporal_config['label_end_time']
    model_update_frequency = temporal_config['model_update_frequency']
    training_as_of_date_frequencies = temporal_config['training_as_of_date_frequencies']
    test_as_of_date_frequencies = temporal_config['test_as_of_date_frequencies']
    max_training_histories = temporal_config['max_training_histories']
    training_label_timespans = temporal_config['label_timespans']
    test_label_timespans = temporal_config['label_timespans']
    test_durations=temporal_config['test_durations']


    chopper = Timechop(
        feature_start_time=feature_start_time,
        feature_end_time=feature_end_time,
        label_start_time=label_start_time,
        label_end_time=label_end_time,
        model_update_frequency=model_update_frequency,
        training_as_of_date_frequencies=training_as_of_date_frequencies,
        max_training_histories=max_training_histories,
        training_label_timespans=training_label_timespans,
        test_as_of_date_frequencies=test_as_of_date_frequencies,
        test_durations=test_durations,
        test_label_timespans=test_label_timespans
    )

    dateTimeObj = datetime.now()
    timestr = dateTimeObj.strftime('%Y%m%d%H%M')
    user = getpass.getuser()
    cf = ntpath.basename(config_file)[0:10]
    data_folder = '/mnt/data/experiment_data/'
    save_path = os.path.join(data_folder, 'triage', 'elsal', 'timechops', '{}_{}_{}.png'.format(user, timestr, cf))
    

    visualize_chops(
        chopper=chopper,
        show_as_of_times=True,
        show_boundaries=True,
        save_target=save_path
    )

if __name__ == '__main__':
    config_file = sys.argv[1]  
    n_jobs = sys.argv[2]
    logging.info('Running the experiment from {} with {} processes'.format(
            config_file, n_jobs))

    run_exp(
        config_file, 
        plot_timechops=False, 
        run_exp=True,
        use_s3=False,
        s3_path='s3://dsapp-cmu-research/triage/el_salvador',
        n_jobs=int(n_jobs)
    )

