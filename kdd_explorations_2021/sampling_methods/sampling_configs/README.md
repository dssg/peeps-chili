## Writing sampling config file
###### Example

* `experiment_hash`: Experiment hash corresponding to Triage run on non-sampled matrices
* `model_group_id`: A base model group id, ideal to put the dummy model group here.
* `multiplier`: Multiplier for entity_id - the resulting entity_ids are used for sampled matrices
* `base_dir`: Base directory from where original train and test matrices should be read.
* `orig_data_name`: Directory where matrices are stored. base_dir+'/'+orig_data_name gives you the dir for the matrices
* `demo_table`: Demographics table containing information about entities
* `orig_demo_col`: Column for demographics
* `demo_col`: Name of demographic column
* `demo_permutations`: Different values demographic column can obtain
* `label`: Column name for label information in matrix. (Y=0/1)