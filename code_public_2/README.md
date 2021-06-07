# This repository has the following structure:
sampling_methods: Contains code for sampling the dataset in different configurations.
``python test_sampling.py test_sampling_config.yaml``
triage: Contains all the code to run [triage](https://github.com/dssg/triage) which takes as an input triage config file.

## Experiment Running Details

#### Sampling Runs [Pre-Processing]

1. Run `triage` on the dataset with the original configuration.
2. Run Sampling to generate sampled matrices.
3. For each sampling config, run `triage` with `replace=False`; change `model_comment` and set project_dir as the directory of the sampled matrices.

#### No Protected [Pre-Processing]
1. In the original `triage` config file, remove the features that contribute to demographics information.
2. Run `triage` with a different project dir and `model_comment`.

#### Zafar et. al. [Post-Processing]
1. See `zafar_methods` folder
2. The `fair-classification` folder is a dependency for running this method.

#### Decoupled []
1. Create two configs from the original config. Modify the `cohort` information such that each config runs only on subset of the entities belonging to only one demographic group.
2. Run `triage`

#### Fully Coupled []



For any of the above methods, you can run RecallAdjuster to balance equity while minimizing the compromise in precision (See each project's folder in `RecallAdjuster` for details)
