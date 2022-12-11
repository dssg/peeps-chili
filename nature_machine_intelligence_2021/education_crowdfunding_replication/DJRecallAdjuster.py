import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

import os
import yaml
import sqlalchemy


from itertools import permutations
from jinja2 import Template
import dateparser


from ohio.ext.numpy import pg_copy_to_table

try:
    import matplotlib.pyplot as plt
except (ImportError, RuntimeError) as e:
    print("matplotlib import error -- you are likely using the terminal, so plot() functions will not be available")
    pass


ENTITY_DEMO_FILES = {
    'joco': {
        'sql_tmpl': 'joco_entity_demos.sql.tmpl',
        'check_sql': """
            WITH all_matches AS (
                SELECT COUNT(DISTINCT ((mg.model_config->'matchdatetime')::VARCHAR)::TIMESTAMP) AS num_match
                FROM tmp_bias_models
                JOIN model_metadata.model_groups mg USING(model_group_id)
            )
            SELECT num_match = 1 AS pass_check
            FROM all_matches
        """
        }
}



class RecallAdjuster(object):
    def __init__(
        self,
        engine,
        params,
        pause_phases=False, 
        alternate_save_names=[]
        ):
        """
        Arguments:
            engine: 
                An engine for a postgres database
            params:
                Dictionary with following properly defined
                pg_role:
                    Role to use in postgres
                schema:
                    Schema for table creation
                experiment_hashes:
                    A list of strings with triage experiments to include
                date_pairs:
                    A list of tuples of train_end_times (as strings). The first should be the date to use
                    to make adjustments and the second a future date for evaluation.
                list_sizes:
                    A list of integers, the sizes of lists to generate.
                entity_demos:
                    Either a table name (in format "schema.table_name") or "joco" to use
                    JoCo-specific code to create this table on the fly.
                demo_col:
                    Column name containing demographic data on which to make adjustments
                sample_weights:
                    Optional dictionary of demo values to weight for subsampling. Excluded demo values
                    will be included in their entirely with no sampling. Weights should be a value
                    between 0 and 1.0, reflecting the fraction of that demographic to include.
                decoupled_experiments:
                    Optional list of tuples of (experiment_hash, demo_value) that identify decoupled
                    experiments with models run using data only from each subgroup. Data from these
                    experiments will be used only to create a composite that allows EITHER the decoupled
                    or full models to be used for each subgroup. Multiple experiments can be specified
                    for a given demo_value, but all demo_values must be included.
                decoupled_entity_demos:
                    Optional "schema.table_name" for a separate entity_demos table to be used for the decoupled
                    experiments, for instance in cases where entity_ids may differ between modeling runs
                    such as is the case with JoCo matches. If specified, must be pre-computed.
                entity_demos:
                    e.g: bias_working.entity_demos
                weights:
                    Weighting scheme for multi adjustment as a list of fractions
                date_list:
                    List of all dates in increasing order
                min_separations:
                    The minimum time we must go back given a future_train_end_time to know we have that data and label set at prediction time of future_train_end_time. If not stated we assume 2
            pause_phases:
                True if you want a break after each phase requiring user input to continue
        """

        # store parameters
        self.engine = engine.connect()
        self.params = params
        
        self.params['date_weights'] = self.get_date_weights()
        self.params['date_weight_case_str'] = self.get_weight_case_str()
        self.params['date_weight_past_train_end_time_case_str'] = self.get_weight_past_train_end_time_case_str()
        

        # check consistency of date pairs
        self.validate_dates()

        # create a few temporary tables we'll need for calculations
        sql = Template(open('recall_adjustment_pre.sql.tmpl', 'r').read()).render(**self.params)
        self.engine.execute(sql)
        self.engine.execute("COMMIT")
        
        if pause_phases:
            input(f"Date Pair: {self.params['date_pairs']} pre sql done")

        entity_demos = self.params['entity_demos']
        if entity_demos.find('.') > -1:
            self.params['entity_demos'] = entity_demos
        elif entity_demos in ENTITY_DEMO_FILES.keys():
            self.params['entity_demos'] = self.create_entity_demos(entity_demos)
        else:
            raise ValueError('Error: entity_demos must be either `schema.table_name` OR one of (%s)' % ', '.join(ENTITY_DEMO_FILES.keys()))

        # calculate demo values for general use, ordered by frequency
        sql = "SELECT %s, COUNT(*) AS num FROM %s GROUP BY 1 ORDER BY 2 DESC" % (self.params['demo_col'], self.params['entity_demos'])
        res = self.engine.execute(sql).fetchall()
        self.params['demo_values'] = [r[0] for r in res]
        self.params['demo_permutations'] = list(permutations(self.params['demo_values'], 2))
        

        # pre-calculate the results for all models, date pairs
        sql = Template(open('recall_adjustment_verbose.sql.tmpl', 'r').read()).render(**self.params)
        self.engine.execute(sql)
        self.engine.execute("COMMIT")
        
        if pause_phases:
            input(f"Date Pair: {self.params['date_pairs']} Adjustment Done")

        # store the results to dataframes for subsequent plotting and analysis
        sql = 'SELECT * FROM %s.model_adjustment_results_%s' % (self.params['schema'], self.params['demo_col'])
        self.adjustment_results = pd.read_sql(sql, self.engine)

        sql = 'SELECT * FROM %s.composite_results_%s' % (self.params['schema'], self.params['demo_col'])
        self.composite_results = pd.read_sql(sql, self.engine)
        
        for save_name in alternate_save_names:
            schema = self.params['schema'] 
            demo_col = self.params["demo_col"]
            sql = f"DROP TABLE IF EXISTS {schema}.{save_name}; CREATE {schema}.{save_name} AS SELECT * FROM {schema}.model_adjustment_results_{demo_col}"

        self.engine.close()

    
    def get_date_weights(self):
        weights = self.params['weights']
        date_list = self.params['date_list']
        min_separation = self.params.get('min_separations', 2)
        date_weights = {}
        i_lim = len(weights)
        assert i_lim >= min_separation
        for i, date in enumerate(date_list):
            if i < i_lim + min_separation - 1:
                date_weights[date] = {date_list[0]: 1.0, "past_train_end_time": date_list[0]}
            else:
                base = i - min_separation # No matter what latest time we can use is 2 before current month
                d = {}
                for j in range(len(weights)):
                    d[date_list[base - j]] = weights[j]
                d["past_train_end_time"] = date_list[base] # Should be the most recent date used to compute adjustment
                date_weights[date] = d
        return date_weights

    
    
    def get_weight_past_train_end_time_case_str(self):
        date_weights = self.params['date_weights']
        date_list = self.params['date_list']
        if len(date_weights) == 0:
            return f"'{date_list[0]}'::TIMESTAMP"
        s = "CASE"
        for future_train_end_time in date_weights:
            past_train_end_time = date_weights[future_train_end_time]["past_train_end_time"]
            s += f" WHEN future_train_end_time = '{future_train_end_time}' THEN '{past_train_end_time}'::TIMESTAMP"
        s += f" ELSE '{date_list[0]}'::TIMESTAMP END"
        return s
    
    def get_weight_case_str(self):
        date_weights = self.params['date_weights']
        if len(date_weights) == 0:
            return "0"
        s = "CASE"
        for future_train_end_time in date_weights:
            for train_end_time in date_weights[future_train_end_time]:
                if train_end_time == "past_train_end_time":
                    pass
                else:
                    w = date_weights[future_train_end_time][train_end_time]
                    s += f" WHEN future_train_end_time = '{future_train_end_time}' AND train_end_time = '{train_end_time}' THEN {w} "
        s += "ELSE 0 END"
        return s

    
    def ensure_all_demos(self, check_demos):
        all_demos = set(self.params['demo_values'])
        check_demos = set(check_demos)
        if all_demos - check_demos:
            raise ValueError('Error: demo values not found in decoupled_experiments - %s' % (all_demos - check_demos))
        if check_demos - all_demos:
            raise ValueError('Error: demo values specified in decoupled_experiments not found in data - ' % (check_demos - all_demos))


    def validate_dates(self):
        for past, future in self.params['date_pairs']:
            if dateparser.parse(past) > dateparser.parse(future):
                raise ValueError('Error! Cannot validate on the past. %s should be no earlier than %s.' % (future, past))


    def do_bootstrap(self):
        # FIXME: Right now bootstrap sampling relies on duplicate entity_id values in tmp_bias_sample
        #        not creating downstream issues. This should be true currently, but is a bit of a
        #        risky assumption should someone change downstream code to either de-dupe on entity
        #        or potentially introduce dupes elsewhere that could result in a many-to-many join...
        self.engine.execute("DROP TABLE IF EXISTS tmp_bias_sample;")
        self.engine.execute("""
            CREATE LOCAL TEMPORARY TABLE tmp_bias_sample (
                    entity_id INT,
                    as_of_date DATE
                ) ON COMMIT PRESERVE ROWS;
            """)
        self.engine.execute("COMMIT;")

        # gross nested for loop that could probably be done away with, but sizes should be pretty small, so meh...
        for as_of_date in set([dt for dt_pair in self.params['date_pairs'] for dt in dt_pair]):
            tot_size = self.engine.execute("SELECT COUNT(DISTINCT entity_id) FROM %s WHERE as_of_date='%s'::DATE" % (self.params['entity_demos'], as_of_date)).fetchall()[0][0]
            for demo_value, demo_frac in self.params['bootstrap_weights'].items():
                demo_size = round(tot_size*demo_frac)
                all_entities = self.engine.execute("""
                    SELECT DISTINCT entity_id
                    FROM {entity_demos}
                    WHERE {demo_col} = '{demo_value}'
                        AND as_of_date = '{as_of_date}'::DATE
                    ;
                    """.format(
                        entity_demos=self.params['entity_demos'], 
                        demo_col=self.params['demo_col'],
                        demo_value=demo_value,
                        as_of_date=as_of_date
                        ))
                all_entities = np.array([row[0] for row in all_entities])
                bs_entities = np.random.choice(all_entities, size=demo_size, replace=True)
                bs_entities = np.array([[e, as_of_date] for e in bs_entities])
                pg_copy_to_table(bs_entities, 'tmp_bias_sample', self.engine, columns=['entity_id', 'as_of_date'], fmt=('%s', '%s'))
                self.engine.execute("COMMIT;")
        self.engine.execute("CREATE INDEX ON tmp_bias_sample(entity_id, as_of_date);")


    def create_entity_demos(self, entity_demos):
        sql_file = ENTITY_DEMO_FILES[entity_demos]['sql_tmpl']
        sql = Template(open(sql_file, 'r').read()).render(**self.params)
        self.engine.execute(sql)
        self.engine.execute("COMMIT")

        # consistency check:
        check_sql = ENTITY_DEMO_FILES[entity_demos]['check_sql']
        if not self.engine.execute(check_sql).fetchall()[0][0]:
            raise RuntimeError('Entity Demos failed consistency check:\n %s' % check_sql)

        return '%s.entity_demos' % self.params['schema']


    def plot(
        self,
        plot_type='shift',
        recall_ratio='largest',
        date_pair=None,
        list_size=None,
        metric='precision@',
        ax=None
        ):
        """
        Arguments:
            plot_type:
                One of `before`, `after`, `shift` (default)
            recall_ratio:
                May be `largest` (default) to plot against the largest recall ratio, `all_demos` to plot
                all pairwise ratios across demo values, or `{demo1}_to_{demo2}` to plot a
                spefic ratio between two given demo values
            date_pair:
                The tuple representing the past and future train_end_times to use for the plot
                If not specified, the latest pair will be used
            list_size:
                The list size to use for plotting (If unspecified, the largest value will be used)
            metric:
                The metric for plotting, currently only 'precision@' is supported
            ax:
                Optionally pass an axes object for the plot to use
        Returns:
            ax_dict:
                Dictionary mapping recall_ratio to the axis used by the plot, to allow further
                modification of display parameters
        """

        # FIXME: remove print statements in favor of labels on the figures!

        if date_pair is None:
            date_pair = sorted(self.params['date_pairs'], key=lambda x: (x[1], x[0]), reverse=True)[0]

        if list_size is None:
            list_size = sorted(self.params['list_sizes'], reverse=True)[0]

        if metric != 'precision@':
            return ValueError("Currently `precision@` is the only supported metric!")

        if recall_ratio == 'all_demos':
            # just print these once...
            print("Date Pair: %s" % str(date_pair))
            print("List Size: %s" % list_size)
            print("Metric: %s%s_abs" % (metric, list_size))

            # TODO: Could probably make these a small multiples grid?
            ax_dict = {}
            # set up a figure for the plots
            num_plots = len(self.params['demo_permutations'])
            figsize = plt.rcParams['figure.figsize'].copy()
            figsize[1] = figsize[1]*num_plots
            _, ax = plt.subplots(num_plots, 1, sharex=False, sharey=False, figsize=figsize)

            for i, (demo1, demo2) in enumerate(self.params['demo_permutations']):
                recall_ratio = '%s_to_%s' % (demo1, demo2)
                ax_dict.update(self.plot(plot_type, recall_ratio, date_pair, list_size, metric, ax=ax[i]))
            plt.tight_layout(h_pad=1.1, w_pad=1.1)
            return ax_dict

        elif recall_ratio == 'largest':
            plot_ratio = 'max_recall_ratio'
            ylabel = 'largest recall ratio'

        else:
            plot_ratio = ('recall_%s' % recall_ratio).lower()
            ylabel = 'recall ratio: %s' % recall_ratio.replace('_to_', '/')

        if plot_type == 'shift':
            plot_title = 'Equity and Efficiency Movement'
        elif plot_type == 'after':
            plot_title = 'Equity and Efficiency After Adjustment'
        elif plot_type == 'before':
            plot_title = 'Equity and Efficiency Before Adjustment'

        if ax is not None:
            plt.sca(ax)
        else:
            print("Date Pair: %s" % str(date_pair))
            print("List Size: %s" % list_size)
            print("Metric: %s%s_abs" % (metric, list_size))
            _, ax = plt.subplots()

        # subset the adjustment results dataframe to the current parameters
        sub_df = self.adjustment_results.loc[
            (self.adjustment_results['list_size'] == list_size)
            &
            (self.adjustment_results['metric'] == metric)
            &
            (self.adjustment_results['train_end_time'] == dateparser.parse(date_pair[1]))
            &
            (self.adjustment_results['past_train_end_time'] == dateparser.parse(date_pair[0]))
        ,
        ['base_value', 'base_%s' % plot_ratio, 'adj_value', 'adj_%s' % plot_ratio]
        ]

        ylim = 1.1*max(
            sub_df['base_%s' % plot_ratio].max(),
            sub_df['adj_%s' % plot_ratio].max(),
            1.0
            )

        xmin = 0.8*min(
            sub_df['base_value'].min(),
            sub_df['adj_value'].min()
            )
        xmax = 1.1*max(
            sub_df['base_value'].max(),
            sub_df['adj_value'].max()
            )

        arr = sub_df.values

        # plot a reference line at y = 1 and the desired points
        plt.plot((0,1),(1,1),'k-', zorder=0)
        for x0, y0, x1, y1 in arr:
            if plot_type == 'shift':
                plt.plot((x0,x1), (y0,y1), 'k-', alpha=0.5)
            if plot_type in ('before', 'shift'):
                plt.plot(x0, y0, color='C0', marker='o')
            if plot_type in ('after', 'shift'):
                plt.plot(x1, y1, color='C1', marker='o')

        # For after and shift plots, add the composite point as a red diamond
        if plot_type in ('after', 'shift'):

            comp_arr = self.composite_results.loc[
                (self.composite_results['list_size'] == list_size)
                &
                (self.composite_results['metric'] == metric)
                &
                (self.composite_results['train_end_time'] == dateparser.parse(date_pair[1]))
                &
                (self.composite_results['past_train_end_time'] == dateparser.parse(date_pair[0]))
            ,
            ['value', plot_ratio]
            ].values

            if len(comp_arr) > 1:
                raise ValueError("Uniqueness error! Check composite results for duplicate results.")

            plt.plot(comp_arr[0][0], comp_arr[0][1], marker='D', color='red', markersize=7)

        ax = plt.gca()
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((0.0, ylim))
        ax.set_xlabel('%s%s_abs' % (metric, list_size))
        ax.set_ylabel(ylabel)
        ax.set_title(plot_title)

        # plt.show()

        return {recall_ratio: ax}

    
def education_ra_procedure(weights=[0.99, 0.01], alternate_save_names=[], engine_donors=None, config=None, pause_phases=False):
    """
    Because of the size of the data, we're going to do this iteratively over subsets of validation set dates to avoid running into memory issues, but depending on your dataset and database server, you could instead simply run the `RecallAdjuster` once with the full set of date pairs.
    
    date_pairs: The code needs a "previous" validation set to learn the group-specific thresholds to equalize recall on the "current" validation set. Additionally, for every date used as a "previous" validation set, we include a pair with this set as both the "previous" and "current" date (to allow for selecting a model based on post-adjustment performance on the previous set)
    pg_role: Allows you to set a different role in postgres if needed, but generally will be the same as your postgres user
    schema: We'll use `bias_working` for the intermediate results of each iteration, but then will collect all of these into the `bias_results` schema
    experiment_hashes: Triage tracks runs of a grid of models via an "experiment" object, identified by this hash. The one coded here is for the set of models described in the study
    list_sizes: The overall "top k" size(s) to consider (can be a list of multiple, but if so, you'll need to be careful to modify the results query above to choose just one at a time)
    entity_demos: This is a postgres table containing a lookup between entities (here, projects), dates, and demographics of interest for bias analysis. Here, `bias_working.entity_demos` contains the school poverty levels determined from the project data.
    demo_col: The specific column of interest for the bias analysis. Here, `plevel` is the school poverty level.
    """
    if engine_donors is None or config is None:
        with open('db_profile.yaml') as fd:
            config = yaml.full_load(fd)
            dburl = sqlalchemy.engine.url.URL(
                "postgresql",
                host=config["host"],
                username=config["user"],
                database=config["db"],
                password=config["pass"],
                port=config["port"],
            )
            engine_donors = sqlalchemy.create_engine(dburl, poolclass=sqlalchemy.pool.QueuePool)
        
    engine_donors.execute('TRUNCATE TABLE bias_results.composite_results_plevel;')
    engine_donors.execute('TRUNCATE TABLE bias_results.model_adjustment_results_plevel;')
    engine_donors.execute('TRUNCATE TABLE bias_working.model_adjustment_group_k_plevel;')
    engine_donors.execute('COMMIT;')
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

    date_list = ['2011-03-01', '2011-05-01', '2011-07-01', '2011-09-01', '2011-11-01', '2012-01-01', '2012-03-01', '2012-05-01', '2012-07-01', '2012-09-01', '2012-11-01', '2013-01-01']
    
    for dp_idx in range(10):
        date_pairs = [ date_pairs_all[2*dp_idx], date_pairs_all[2*dp_idx+1] ]
        print(date_pairs)
        params = {}
        params['pg_role'] = config["user"]
        params['schema'] = 'bias_working'
        experiment_hashes = ['a33cbdb3208b0df5f4286237a6dbcf8f']
        params['experiment_hashes'] = experiment_hashes
        if isinstance(date_pairs[0], str):
            date_pairs = [date_pairs]
        params['date_pairs'] = date_pairs
        params['date_list'] = date_list
        params['weights'] = weights
        params['list_sizes'] = [1000]
        params['demo_col'] = 'plevel'
        params['subsample'] = False
        params['bootstrap'] = False
        params['entity_demos']='bias_working.entity_demos'


        engine=engine_donors
        ra = RecallAdjuster(engine=engine, params=params, pause_phases=pause_phases, alternate_save_names=alternate_save_names)
        
        engine_donors.execute("""
            INSERT INTO bias_results.model_adjustment_results_plevel 
            SELECT * FROM bias_working.model_adjustment_results_plevel;
        """)
    
        engine_donors.execute("""
            INSERT INTO bias_results.composite_results_plevel 
            SELECT * FROM bias_working.composite_results_plevel;
        """)

        engine_donors.execute("""
            INSERT INTO bias_results.model_adjustment_group_k_plevel 
            SELECT * FROM bias_working.model_adjustment_group_k_plevel gkp WHERE (gkp.model_group_id, gkp.train_end_time, gkp.demo_value, gkp.group_k) NOT IN (SELECT * FROM bias_results.model_adjustment_group_k_plevel)
        """)

        engine_donors.execute("""
            INSERT INTO bias_results.model_multi_adjustment_results_plevel
            SELECT * FROM bias_working.model_multi_adjustment_results_plevel;
        """)

        engine_donors.execute("COMMIT;")

if __name__ == "__main__":
    education_ra_procedure(weights=[0.99, 0.01], alternate_save_names=["save_res_nn_o"])
    #education_ra_procedure(weights=[0.01, 0.99], alternate_save_names=["save_res_o_nn"])
    #education_ra_procedure(weights=[0.5, 0.5], alternate_save_names=["save_res_half"])
    
