#!/bin/bash
#SBATCH --partition=ai2es_v100
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --job-name=hparam_ANN
#SBATCH --mail-user=randychase@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/randychase/slurmouts/R-%x.%j.out
#SBATCH --error=/home/randychase/slurmouts/R-%x.%j.err

#source my python env
source /home/randychase/.bashrc
bash 

cd /ourdisk/hpc/ai2es/randychase/debug/deep_learning/

#activate the tensorflow environment
conda activate different_tf 

#run the python code 
python -u hparams_CNN_weather-mnist.py --logdir="/ourdisk/hpc/ai2es/randychase/boardlogs/weather-mnist/CNN/class/scalar_logs/"