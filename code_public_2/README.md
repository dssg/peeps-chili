# This repository has the following structure:
sampling_methods: Contains code for sampling the dataset in different configurations.
``python test_sampling.py test_sampling_config.yaml``
triage: Contains all the code to run [triage](https://github.com/dssg/triage) which takes as an input triage config file.

## Experiment Running Details

#### Sampling Runs

1. Run `triage` on the dataset with the original configuration.
2. Run Sampling to generate sampled matrices.
3. For each sampling config, run `triage` with `replace=False` and project_dir as the directory of the sampled matrices.

#### No Protected
1. In the original `triage` config file, remove the features that contribute to demographics information.
2. Run `triage`

#### Zafar
1. See `zafar_methods` folder

#### Decoupled
1. Create two configs from the original config. Modify the `cohort` information such that each config runs only on subset of the entities belonging to only one demographic group.
2. Run `triage`


For any of the above methods, you can run RecallAdjuster to balance equity while minimizing the compromise in precision (See each project's folder in `RecallAdjuster` for details)
