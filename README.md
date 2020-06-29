# Fairness-Accuracy Trade-Offs in ML for Public Policy
This repository contains code relating to our ongoing work explorating the trade-offs between fairness and accuracy in machine learning models developed to support decision-making in public policy contexts.

For each context, modeling was performed with our open-sourced machine learning pipeline toolkit, [triage](https://github.com/dssg/triage). Although the data for several of these projects is confidential and not publicly available, this repository includes our `triage` configuration files (specifying features and model/hyperparameter grids) for all projects as well as the code used for bias mitigation and analysis of trade-offs. The main functionality for bias mitigation is provided in `RecallAdjuster.py` (at the moment, this assumes model results are in the form of `triage` output) and analyses are generally in a set of `jupyter` notebooks in each project directory. The bias mitigation here extends on methods we described recently at [FAT* 2020](https://arxiv.org/abs/2001.09233). Each project is described briefly below:

### Inmate Mental Health
The Inmate Mental Health project focuses on breaking the cycle of incarceration in Johnson County, KS, by proactive outreach from their Mental Health Center's Mobile Crisis Response Team to individuals with a history of incarceration and mental health need and at risk of returning to jail. Early results from this work was presented at [ACM COMPASS 2018](https://dl.acm.org/citation.cfm?id=3209869) and code for this analysis can be found in [code/joco](code/joco).

### Housing Safety
The Housing Safety project involved helping the Code Enforcement Office in the City of San Jose prioritize inspections of multiple housing properties (such as apartment buildings) to identify health and safety violations that might put their tenants at risk. Some more details about the approach and initial results of this work can be found in [this blog post](http://www.dssgfellowship.org/2017/07/14/data-driven-inspections-for-safer-housing-in-san-jose-california/) and code for this analysis can be found in [code/housing_safety](code/housing_safety).

### Student Outcomes
In the Student Outcomes project, we partnered with El Salvadar to help them target interventions for students at risk of dropping out of school each year. The repository from this project was made publicly available and contains a detailed overview of that work [here](https://github.com/dssg/El_Salvador_mined_education) and the code for the fairness-accuracy trade-off investigations can be found in [code/el_salvador](code/el_salvador).

### Eduction Crowdfunding
Because the data from these other projects cannot be publicly released, we have also been investigating these trade-offs in the context of a project based around data [made public by DonorsChoose in 2014](https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/data) to provide an example that can be more readily reproduced by other researchers. Code relating to this case study can be found in [code/education_crowdfunding](code/education_crowdfunding).

