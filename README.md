# Fairness-Accuracy Trade-Offs in ML for Public Policy
[![DOI](https://zenodo.org/badge/254551159.svg)](https://zenodo.org/badge/latestdoi/254551159)

This repository contains code relating to our ongoing work explorating the trade-offs between fairness and accuracy in machine learning models developed to support decision-making in public policy contexts.

For each context, modeling was performed with our open-sourced machine learning pipeline toolkit, [triage](https://github.com/dssg/triage). Although the data for several of these projects is confidential and not publicly available, this repository includes our `triage` configuration files (specifying features and model/hyperparameter grids) for all projects as well as the code used for bias mitigation and analysis of trade-offs. The main functionality for bias mitigation is provided in `RecallAdjuster.py` (at the moment, this assumes model results are in the form of `triage` output) and analyses are generally in a set of `jupyter` notebooks in each project directory. Note that in the process of this work, we have made slight adjustments to the `RecallAdjuster` code to accomodate the data in each context, so be sure to use the correct version if working on one of these projects (consolidating these versions is tracked in issue #7).

The bias mitigation here extends on methods we described recently at [FAT* 2020](https://arxiv.org/abs/2001.09233). Addittionally, we recently developed a [tutorial](https://dssg.github.io/fairness_tutorial/) around improving machine learning fairness and a simplified application can be found in [this interactive colab notebook](https://colab.research.google.com/github/dssg/fairness_tutorial/blob/master/notebooks/bias_reduction.ipynb) which is a good starting point.

Each project is described briefly below:

### Inmate Mental Health
The Inmate Mental Health project focuses on breaking the cycle of incarceration in Johnson County, KS, by proactive outreach from their Mental Health Center's Mobile Crisis Response Team to individuals with a history of incarceration and mental health need and at risk of returning to jail. Early results from this work was presented at [ACM COMPASS 2018](https://dl.acm.org/citation.cfm?id=3209869) and code for this analysis can be found in [code/joco](code/joco).

### Housing Safety
The Housing Safety project involved helping the Code Enforcement Office in the City of San Jose prioritize inspections of multiple housing properties (such as apartment buildings) to identify health and safety violations that might put their tenants at risk. Some more details about the approach and initial results of this work can be found in [this blog post](http://www.dssgfellowship.org/2017/07/14/data-driven-inspections-for-safer-housing-in-san-jose-california/) and code for this analysis can be found in [code/housing_safety](code/housing_safety).

### Student Outcomes
In the Student Outcomes project, we partnered with El Salvadar to help them target interventions for students at risk of dropping out of school each year. The repository from this project was made publicly available and contains a detailed overview of that work [here](https://github.com/dssg/El_Salvador_mined_education) and the code for the fairness-accuracy trade-off investigations can be found in [code/el_salvador](code/el_salvador).

### Education Crowdfunding
Because the data from these other projects cannot be publicly released, we have also been investigating these trade-offs in the context of a project based around data [made public by DonorsChoose in 2014](https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/data) to provide an example that can be more readily reproduced by other researchers. Code relating to this case study can be found in [code/education_crowdfunding](code/education_crowdfunding).


## Replication with Public Data

Although several of the datasets used for this work contain sensitive information and were made available under data use agreements that don't allow for public release, data from the Education Crowdfunding use case is publicly available. Here we provide three options for replicating the analysis from our study:

### Option 1: Colab Notebook (simple example with little setup time)
For a simple application of the methods discussed here on this dataset, take a look at [this interactive colab notebook](https://colab.research.google.com/github/dssg/fairness_tutorial/blob/master/notebooks/bias_reduction.ipynb), which we developed for part of our [FairML tutorial](https://dssg.github.io/fairness_tutorial/) presented at KDD 2020 and AAAI 2021.

### Option 2: Replicating Bias Analysis with Existing Models (step-by-step notebook)
To facilitate replicating the full results from the Education Crowdfunding setting presented in the study, we have made an extract of our database publicly available on S3. This extract contains the results of our model runs as well as the bias adjustment analysis presented in the study and can easily be used either to replicate our figures or re-run the bias analysis using a step-by-step jupyter notebook in the [/code/education_crowdfunding_replication](/code/education_crowdfunding_replication) directory -- see the [README](/code/education_crowdfunding_replication/README.md) in that directory for instructions on downloading the database dump and getting setup. The extract also contains the raw data from DonorsChoose, so could be used as a starting point for re-running or adding to the model grid as well.

Note that you'll need a postgres server (version 11.10 or above) with around 300 GB of free disk space to load the data extract as well as machine running python 3.7 or higher for the analysis.

### Option 3: Rerunning Models and Bias Analysis from Scratch
If you would like to rerun the models themselves in order to recreate the results starting from the Education Crowdfunding dataset itself, this can be achieved with the following steps:
1. Follow the instructions from the [dssg/donors-choose](https://github.com/dssg/donors-choose) github repo for obtaining and transforming the [DonorsChoose KDD cup 2014 dataset](https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/data) and running a grid of machine learning models
  - The data should be loaded into a postgresql database (we have used postgres 11.10 in our work here) and the machine learning and disparity mitigating code here works with python 3.7 or higher
  - Modeling makes use of our machine learning pipeline, [triage](https://github.com/dssg/donors-choose), to run sklearn models. See the [donors-choose requirements.txt](https://github.com/dssg/donors-choose/blob/master/requirements.txt) and [triage requirements](https://github.com/dssg/triage/tree/master/requirement) for specific versions. Following the installation instructions from the donors-choose repository will install all the necessary dependencies and should take less than an hour.
  - This will produce a set of trained model objects stored on disk as well as predictions and validation results stored into the postgres database
2. Install the requirements specific to the bias mitigation code with `pip install -r /code/education_crowdfunding/requirements.txt`
3. Start a jupyter notebook server and copy the files from [/code/education_crowdfunding](/code/education_crowdfunding) into your notebook directory
4. Follow the steps in [/code/education_crowdfunding/20200612_dc_figures.ipynb](/code/education_crowdfunding/20200612_dc_figures.ipynb) to run the bias mitigation and model selection analysis

The modeling and analysis here have been performed on a server running Ubuntu 18.04, but should run on most linux-based systems. We would recommend running the models on a reasonably well-provisioned server, but on a typical desktop these could probably be expected to complete in 1-2 days. The bias mitigation and model selection analysis would likely require 30-90 minutes on a typical desktop.


