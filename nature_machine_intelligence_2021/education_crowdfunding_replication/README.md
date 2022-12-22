# Replicating Education Crowdfunding Results

The code in this directory provides step-by-step instructions for replicating the results from our recent study of fairness-accuracy trade-offs on the Education Crowdfunding dataset. To make reproducing the bias analysis easier as well as provide more visibility into our results, this it starts off from a point where the public DonorsChoose dataset has already been injested into a postgres database and a grid of models (described in Supplementary Table 2 of our study) has already been run. We've made available a database dump with both the underlying data and the results of our model runs (see below for instructions on getting it set up).

You can also re-run (or add to) the model grid starting from the public DonorsChoose data using our open-source machine learning pipeline `triage`, which makes it easier to run any model with a `scikit-learn`-style interface over temporal validation sets in a "top k" setting. See this repo's top-level readme file for some tips on getting set up as well as pointers to a repo with an example of running triage using these data.

## Requirements

You'll need a postgres server (version 11.10 or above) with around 300 GB of free disk space to load the data extract as well as machine running python 3.7 or higher for the analysis.

## Getting Set Up

1. Install the python requirements on your machine by running `pip install -r requirements.txt` from this directory
2. Download the database dump file (note that this file is about 16 GB compressed, so may take some time to download depending on your connection): https://dsapp-public-data-migrated.s3.amazonaws.com/education_crowdfunding_replication.dmp
3. Create a database on your postgres server to load the extract: `CREATE DATABASE education_crowdfunding`
4. Load the downloaded extract into your database with (filling in your server details):
```
pg_restore -h {POSTGRES_HOST} -p {POSTGRES_PORT} -d education_crowdfunding -U {POSTGRES_USER} -O -j 8 education_crowdfunding_replication.dmp
```
5. Copy the `db_profile_template.txt` to `db_profile.yaml` in this directory and fill in your credentials in the file
6. Start a jupyter notebook server and open [education_crowdfunding_replication.ipynb](education_crowdfunding_replication.ipynb) in your browser
7. Follow the instructions in the notebook for reproducing the figures from the study or re-running the bias analysis. The notebook also provides some notes on further exploring the data and results now that you have it loaded as well.
