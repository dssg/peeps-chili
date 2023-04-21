# Replicating Multiple Timestep Bias Adjustment and Bias Adjustment Analysis results

The code in this directory provides step-by-step instructions for replicating the results from our recent study of fairness-accuracy trade-offs on the Education Crowdfunding, Joco, Housing Safety and El Salvador dataset. 


## Requirements

You'll need a postgres server (version 11.10 or above) with around 300 GB of free disk space to load the data extract as well as machine running python 3.7 or higher for the analysis.

## Getting Set Up

1. Install the python requirements on your machine by running `pip install -r requirements.txt` from this directory
2. Go to  peeps-chili/config and fill in credentials in the file `db_default_profile`
3. Start a jupyter notebook server and open the desired notebook (general_notebook for multi timestep adjustment results, entity_analysis and single_model_analysis for entity and single model cases) in your browser
4. Follow the instructions in the notebook for reproducing the figures from the study or re-running the bias analysis. The notebook also provides some notes on further exploring the data and results now that you have it loaded as well.


## Changing Run Information
Each dataset has a directory with a config file inside it. See DJRecallAdjuster.py to get a sense of what options the config file takes in. To change the dataset run in the notebooks you need to change the `database_directory` variable (often on line 2) to the appropriate directory name, and ensure the config file is set up properly. 
