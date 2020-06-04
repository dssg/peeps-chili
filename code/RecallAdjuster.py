from itertools import permutations
from jinja2 import Template
import dateparser
import pandas as pd
import numpy as np
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
        },
    'donors': {
        'sql_tmpl': 'donors_entity_demos.sql.tmpl',
        'check_sql': """
            select 1=1 AS pass_check
        """,
        'check_sql2': """
            select num_match=1 AS pass_check;
            WITH all_matches AS (
                SELECT COUNT(DISTINCT ((mg.model_config->'matchdatetime')::VARCHAR)::TIMESTAMP) AS num_match
                FROM tmp_bias_models
                JOIN model_metadata.model_groups mg USING(model_group_id)
            )
            SELECT num_match=1 AS pass_check
            FROM all_matches
        """
        },
    'elsal': {
        'sql_tmpl': 'elsal_entity_demos.sql.tmpl',
        'check_sql': """
            select 1=1 AS pass_check
        """
        }
}

class RecallAdjuster(object):
    def __init__(
        self,
        engine,
        pg_role,
        schema,
        experiment_hashes,
        date_pairs,
        list_sizes,
        entity_demos,
        demo_col,
        sample_weights=None,
        bootstrap_weights=None,
        decoupled_experiments=None,
        decoupled_entity_demos=None,
        model_group_ids = None,
        dataset = None,
        demo_values = None
        ):
        """
        Arguments:
            engine: 
                An engine for a postgres database
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
            model_group_ids:
                Optional model_group_ids for choosing specific model groups to select and run RecallAdjuster on.
                This is an alternative way of selecting models in addition to specifying experiment_hash.
            dataset:
                Name of the dataset on which it runs. Specifying the name can allow for running scripts tuned for
                this dataset.
            demo_values:
                Values of the protected attribute we only care about.
        """

        # store parameters
        self.engine = engine.connect()
        self.params = {}
        self.params['pg_role'] = pg_role
        self.params['schema'] = schema
        if isinstance(experiment_hashes, str):
            experiment_hashes = [experiment_hashes]
        self.params['experiment_hashes'] = experiment_hashes
        if isinstance(date_pairs[0], str):
            date_pairs = [date_pairs]
        self.params['date_pairs'] = date_pairs
        if isinstance(list_sizes, int):
            list_sizes = [list_sizes]
        self.params['list_sizes'] = list_sizes
        self.params['demo_col'] = demo_col
        self.params['model_group_ids'] = model_group_ids
        self.params['dataset'] = dataset
        self.params['demo_values'] = demo_values

        # check consistency of date pairs
        self.validate_dates()

        # create a few temporary tables we'll need for calculations
        if(self.params['dataset'] == 'elsal'):
            print("Running ELSal Pre-Recall Adjustment")
            sql = Template(open('elsal_recall_adjustment_pre.sql.tmpl', 'r').read()).render(**self.params)
        else:
            sql = Template(open('recall_adjustment_pre.sql.tmpl', 'r').read()).render(**self.params)

        self.engine.execute(sql)
        self.engine.execute("COMMIT")

        if entity_demos.find('.') > -1:
            self.params['entity_demos'] = entity_demos
            print("Not executing Entity Demos -- picking up from previously built schema.")
        elif entity_demos in ENTITY_DEMO_FILES.keys():
            self.params['entity_demos'] = self.create_entity_demos(entity_demos)
        else:
            raise ValueError('Error: entity_demos must be either `schema.table_name` OR one of (%s)' % ', '.join(ENTITY_DEMO_FILES.keys()))

        # calculate demo values for general use, ordered by frequency
        sql = "SELECT %s, COUNT(*) AS num FROM %s GROUP BY 1 ORDER BY 2 DESC" % (self.params['demo_col'], self.params['entity_demos'])
        res = self.engine.execute(sql).fetchall()
        if(self.params['demo_values']==None):
            print("Setting demo values. Not using pre-set values")
            self.params['demo_values'] = [r[0] for r in res]
        self.params['demo_permutations'] = list(permutations(self.params['demo_values'], 2))

        if sample_weights is not None and bootstrap_weights is not None:
            raise ValueError('Error: may only specify one resampling strategy: sample_weights or bootstrap_weights!')
        elif sample_weights is not None:
            self.params['subsample'] = True
            self.params['bootstrap'] = False
            self.params['sample_weights'] = {d: 1.0 for d in self.params['demo_values']}
            self.params['sample_weights'].update(sample_weights)
        elif bootstrap_weights is not None:
            self.params['subsample'] = False
            self.params['bootstrap'] = True
            self.ensure_all_demos(list(bootstrap_weights.keys()))
            assert(sum(bootstrap_weights.values()) == 1.0)
            self.params['bootstrap_weights'] = bootstrap_weights
            self.do_bootstrap()
        else:
            self.params['subsample'] = False
            self.params['bootstrap'] = False
            print("Not running subsampling OR bootstrap based models")
        if decoupled_experiments:
            # check that all demo_values have coverage in decoupled_experiments
            self.ensure_all_demos([de[1] for de in decoupled_experiments])

            self.params['decoupled_experiments'] = decoupled_experiments

            # default to entity_demos if no decoupled_entity_demos specified
            self.params['decoupled_entity_demos'] = decoupled_entity_demos or self.params['entity_demos']

        # pre-calculate the results for all models, date pairs
        sql = Template(open('recall_adjustment.sql.tmpl', 'r').read()).render(**self.params)
        self.engine.execute(sql)
        self.engine.execute("COMMIT")

        # store the results to dataframes for subsequent plotting and analysis
        sql = 'SELECT * FROM %s.model_adjustment_results_%s' % (self.params['schema'], self.params['demo_col'])
        self.adjustment_results = pd.read_sql(sql, self.engine)

        sql = 'SELECT * FROM %s.composite_results_%s' % (self.params['schema'], self.params['demo_col'])
        self.composite_results = pd.read_sql(sql, self.engine)

        self.engine.close()


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
            metric:s
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
