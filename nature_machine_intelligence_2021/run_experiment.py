import yaml

from sqlalchemy.engine.url import URL
from triage.util.db import create_engine
from sqlalchemy import create_engine as engine_creator
from triage.component.timechop import Timechop
from triage.component.timechop.plotting import visualize_chops
from triage.component.architect.feature_generators import FeatureGenerator
from triage.experiments import MultiCoreExperiment, SingleThreadedExperiment
import logging

from sqlalchemy.pool import NullPool


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh = logging.FileHandler('triage.log', mode='w')
fh.setFormatter(formatter)
logger.addHandler(fh)


database = None
project_path = None


# creating database engine
assert database is not None
assert project_path is not None

with open('../config/db_default_profile.yaml') as fd:
    config = yaml.full_load(fd)
    dburl = sqlalchemy.engine.url.URL(
        "postgresql",
        host=config["host"],
        username=config["user"],
        database=database,
        password=config["pass"],
        port=config["port"],
    )
db_engine = sqlalchemy.create_engine(dburl, poolclass=sqlalchemy.pool.QueuePool)


# loading config file
config_file = 'config.txt'
with open(config_file, 'r') as fin:
    config = json.loads(fin.read())


# creating experiment object

experiment = MultiCoreExperiment(
    config = config,
    db_engine = db_engine,
    project_path = project_path,
    n_processes=2,
    n_db_processes=2,
    replace=False
)

# experiment = SingleThreadedExperiment(
#     config = config,
#     db_engine = db_engine,
#     project_path = 's3://dsapp-education-migrated/donors-choose',
#     replace=True
# )

experiment.validate()
experiment.run()