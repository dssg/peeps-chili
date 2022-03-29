#### Sampling Code

The code can be run by using `python sampling.py` 
Under `all_configs()` you can uncomment the config for which you want to run the sampling. The fields can be enumerated here:

* `mode`: This can be either `under` or `over`
* `ratio_protected`: This is the desired ratio of protected to non-protected. It could be an integer or `original`. Original over here will mean that ratio of protected entities to non-protected entities stay the same as in original matrix.
* `label_dis_within_protected`: This is the ratio of Y=1 to Y=0 within the protected class.
* `label_dis_within_non_protected`: This is the ratio of Y=1 to Y=0 within the non-protected class.