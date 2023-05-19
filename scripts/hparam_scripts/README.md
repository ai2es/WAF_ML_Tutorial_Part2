# Hyperparameter Scripts

We hope it was clear in the tutorial that deep learning you NEED to do some sort of hyperparameter tuning. While tuning can be done in an ad hoc way, a systematic way is often much better. Specifically, we chose to use a stocastic (i.e.,random) manner to search the hyperparameter space. In this folder is some examples of doing this with tensorboard which is tensorflows GUI to help you see how training is going and how to find your 'best' model. This script is a bit 'much' but I have tried to put line by line comments in there. So please reach out if you have ?s. 

An example of actually running the python script is in the 'example_drive_script.sh', this was done on the OU supercomputer so your machine if you sbatch might work well, otherwise just run 

```python -u hparam_script_CNN_class.py --logdir="/ourdisk/hpc/ai2es/randychase/boardlogs/CNN/class/"```

-Cheers, 
Randy 