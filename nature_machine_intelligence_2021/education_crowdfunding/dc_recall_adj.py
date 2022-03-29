#!/usr/bin/env python

import os
import sys
import yaml
import pandas as pd
import sqlalchemy
import RecallAdjuster as ra

def connect(poolclass=sqlalchemy.pool.QueuePool):
    with open('db_default_profile.yaml') as fd:
        config = yaml.load(fd)
        dburl = sqlalchemy.engine.url.URL(
            "postgres",
            host=config["host"],
            username=config["user"],
            database=config["db"],
            password=config["pass"],
            port=config["port"],
        )
        return sqlalchemy.create_engine(dburl, poolclass=poolclass)

conn = connect()

date_pairs_all = [
 ('2011-03-01', '2011-03-01'),
 ('2011-03-01', '2011-07-01'),
 ('2011-05-01', '2011-05-01'),
 ('2011-05-01', '2011-09-01'),
 ('2011-07-01', '2011-07-01'),
 ('2011-07-01', '2011-11-01'),
 ('2011-09-01', '2011-09-01'),
 ('2011-09-01', '2012-01-01'),
 ('2011-11-01', '2011-11-01'),
 ('2011-11-01', '2012-03-01'),
 ('2012-01-01', '2012-01-01'),
 ('2012-01-01', '2012-05-01'),
 ('2012-03-01', '2012-03-01'),
 ('2012-03-01', '2012-07-01'),
 ('2012-05-01', '2012-05-01'),
 ('2012-05-01', '2012-09-01'),
 ('2012-07-01', '2012-07-01'),
 ('2012-07-01', '2012-11-01'),
 ('2012-09-01', '2012-09-01'),
 ('2012-09-01', '2013-01-01')
 ]

dp_idx = int(sys.argv[1]) # starts at 0..9
bias_schema = f'kit_bias_{dp_idx}'
date_pairs = [ date_pairs_all[2*dp_idx], date_pairs_all[2*dp_idx+1] ]

# print(bias_schema)
# print(date_pairs)

myRA = ra.RecallAdjuster(
        engine=conn,
        pg_role='postgres',
        schema=bias_schema,
        experiment_hashes='a33cbdb3208b0df5f4286237a6dbcf8f',
        date_pairs=date_pairs,
        list_sizes=[1000],
        entity_demos='hemank_bias_2way_dates.entity_demos',
        demo_col='plevel'
)





