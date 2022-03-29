# Comparing Fairness Enhancing Methods
The materials in this section of the repository were used in the study:

H. Lamba, R. Ghani\*, and K.T. Rodolfa\*. **An Empirical Comparison of Bias Reduction Methods on Real-World Problems in High-Stakes Policy Settings**. _SIGKDD Explorations_ 23, 69-85. Jun. 2021. [Available in arXiv: 2105.06442](https://arxiv.org/abs/2105.06442)


## This section of the repository has the following structure:
sampling_methods: Contains code for sampling the dataset in different configurations.
``python test_sampling.py test_sampling_config.yaml``
triage: Contains all the code to run [triage](https://github.com/dssg/triage) which takes as an input triage config file.

## Experiment Running Details

#### Resampling the Training Data [Pre-Processing]

1. Run `triage` on the dataset with the original configuration.
2. Run Sampling to generate sampled matrices.
3. For each sampling config, run `triage` with `replace=False`; change `model_comment` and set project_dir as the directory of the sampled matrices.

#### Omitting Sensitive Variables [Pre-Processing]
1. In the original `triage` config file, remove the features that contribute to demographics information.
2. Run `triage` with a different project dir and `model_comment`.

#### Fairness-Constrained Optimization (Zafar et. al.) [In-Processing]
1. Clone `shaycrk/fair-classification` at the `python3` branch into this folder ([link to repo](https://github.com/shaycrk/fair-classification/tree/python3)) -- this is a modification of Zafar's `fair-classification` repository to work with python 3. Note the dependency on `shaycrk/dccp` (which should be handled by installing from the `requirements.txt` from the cloned repo)
2. See `zafar_methods` folder

### Fairness-Aware Model Selection [Post-Processing]
1. Run `triage` to generate a grid of models
2. Follow the analysis in [model_selection/fairness_model_selection.ipynb](model_selection/fairness_model_selection.ipynb) to account for fairness in the model selection process (either by setting a maximum allowable disparity or a maximum allowable loss in accuracy)

#### Decoupled [Post-Processing]
1. Create two configs from the original config. Modify the `cohort` information such that each config runs only on subset of the entities belonging to only one demographic group.
2. Run `triage`

Note: some *very preliminary* code exploring an "ablation" study separating the effects of the decoupling and recall-equalizing score thresholds can be found in the [composite_ablation/](composite_ablation/) directory (this naively assumes the scores from the different models are comparable, which is generally not a reasonable assumption, but may be a common one in applying decoupling approaches).

#### Post-Hoc Score Adjustments [Post-Processing]

For any of the above methods, you can run `RecallAdjuster` to balance equity while minimizing the compromise in precision (See each project's folder in `RecallAdjuster` for details)
