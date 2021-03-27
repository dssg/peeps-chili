import argparse
import logging
import yaml
import json
import datetime
import os

from triage.experiments import MultiCoreExperiment
from triage import create_engine

# 's3://dssg-johnson-county-ddj/triage'
# PROJECT_PATH = '/localdisk'
#PROJECT_PATH = 's3://dsapp-economic-development-migrated/san_jose_housing/triage4'
PROJECT_PATH = 's3://dsapp-cmu-research/BIAS/san_jose_housing/decoupled_under_55k'

def run(config_filename, verbose, replace, predictions, validate_only):
    # configure logging
    log_filename = 'logs/modeling_{}'.format(
        str(datetime.datetime.now()).replace(' ', '_').replace(':', '')
    )
    if verbose:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO
    logging.basicConfig(
        format='%(asctime)s %(process)d %(levelname)s: %(message)s', 
        level=logging_level,
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )

#    config_filename = 'experiment_config'
    features_directory = 'features'

    # load main experiment config
    with open('config/{}.yaml'.format(config_filename)) as f:
        experiment_config = yaml.load(f)

    # load feature configs and update experiment config with their contents
    all_feature_aggregations = []
    for filename in os.listdir('config/{}/'.format(features_directory)):
        with open('config/{}/{}'.format(features_directory, filename)) as f:
            feature_aggregations = yaml.load(f)
            for aggregation in feature_aggregations:
                all_feature_aggregations.append(aggregation)
    experiment_config['feature_aggregations'] = all_feature_aggregations

    with open('config/san_jose_db.json') as f:
        DB_CONFIG = json.load(f)

    db_engine = create_engine(
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['pass']}@{DB_CONFIG['host']}/{DB_CONFIG['db']}"
    )

    experiment = MultiCoreExperiment(
        config=experiment_config,
        db_engine=db_engine,
        project_path=PROJECT_PATH,
        replace=replace,
        n_db_processes=4,
        n_processes=40,
        save_predictions=predictions
    )
    experiment.validate()
    if not validate_only:
        experiment.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run triage pipeline')

    parser.add_argument(
        "-c",
        "--config_filename",
        type=str,
        help="Pass the config filename"
    )

    parser.add_argument("-v", "--verbose", help="Enable debug logging",
                        action="store_true")
    # parser.add_argument("-b", "--baseline", help="Get baseline features",
    #                     action="store_true")
    # parser.add_argument("-p", "--prediction", help="Get prediction matrix",
    #                     action="store_true")


    parser.add_argument(
        "-r",
        "--replace",
        help="If this flag is set, triage will overwrite existing models, matrices, and results",
        action="store_true"
    )
    parser.add_argument(
        "-p",
        "--predictions",
        help="If this flag is set, triage will write predictions to the database",
        action="store_true"
    )
    parser.add_argument(
        "-a",
        "--validateonly",
        help="If this flag is set, triage will only validate",
        action="store_true"
    )

    args = parser.parse_args()

    # if (args.baseline and args.prediction):
    #     raise ValueError('Can only specify one of --prediction or --baseline')

#    run(args.verbose, args.baseline, args.prediction)
    run(args.config_filename, args.verbose, args.replace, args.predictions, args.validateonly)
