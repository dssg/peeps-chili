`zafar_fair_loop.py` runs Zafar based regularization method for all dates for a given dataset. The program takes as an input a config file. 
Examples of config files for different datasets are here:
- zafar_fair_loop_config.yaml
- zafar_loop_donors.yaml
- zafar_loop_elsal.yaml
- zafar_loop_sanjose.yaml

`zafar_post.ipynb` - contains code for running RecallAdjuster. For Zafar method, we don't ideally need to run Recall Adjuster - but it still could be run. 
The Recall Adjuster code in this directory allows to do this.
