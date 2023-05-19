"""
Name: hparam_script_CNN_class.py 
Author: Randy Chase 

Purpose: Do hyperparameter training for a CNN and the subsevir data 

Notes from author: This script is to help users run hyperparameter tuning for a machine learning task. Note this script was used in the machine learning tutorials, 
so the paths to data and data shapes are setup specifically for that dataset, but it should serve as a template to adapt to other things. Note this was
pieced together from here: https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/hparams/hparams_minimal_demo.py. 

The big effort here is to help export data to tensorboard, which helps view training statistics for many models. This saves you time in the end because 
you dont have to then run a seperate 'validation script' to see which models are performing best, this is done inline during training.

Please read the script from the top to bottom. The things you want to define are at the top, then at the bottom is the hyperparameter loop. 

"""

############################### Imports #############################

import os.path
import random
import shutil
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import xarray as xr 
import io
import matplotlib.pyplot as plt 
from datetime import datetime
from tensorboard.plugins.hparams import api as hp
import sys

############################### /Imports #############################

############################### Custom Scripts #############################

#note this is a custom file. Please grab it from the github 
from custom_metrics import MaxCriticalSuccessIndex

############################### /Custom Scripts #############################


#tensorflow check. Only works with tensorflow 2.X
if int(tf.__version__.split(".")[0]) < 2:
    # The tag names emitted for Keras metrics changed from "acc" (in 1.x)
    # to "accuracy" (in 2.x), so this demo does not work properly in
    # TensorFlow 1.x (even with `tf.enable_eager_execution()`).
    raise ImportError("TensorFlow 2.x is required to run this demo.")


############################### Input args #############################
#these are for if you want to control things from the script level. 

#number of hyperparameter choices, default do 100 different configs
flags.DEFINE_integer(
    "num_session_groups",
    100,
    "The approximate number of session groups to create.",
)

#where to put the tensorboard logs
flags.DEFINE_string(
    "logdir",
    "/scratch/randychase/",
    "The directory to write the summary information to.",
)

#this is overwritten later, just ignore. 
flags.DEFINE_integer(
    "summary_freq",
    600,
    "Summaries will be written every n steps, where n is the value of "
    "this flag.",
)

#how many epochs is the ML allowed to train for. 
flags.DEFINE_integer(
    "num_epochs",
    30,
    "Number of epochs per trial.",
)

############################### /Input args #############################

############## Data Shapes ##############

#input shape of your data, (x,y,channel)
INPUT_SHAPE = (48,48,4)
#how many output neruons do you want? Here i only have 1, no lightning or lightning.
OUTPUT_CLASSES = 1

############## /Data Shapes #############


################################ Hyperparameter choices ######################################

#how many convolutions do you want to try out? this says 1,2 and 3 layers of CNNs 
HP_CONV_LAYERS = hp.HParam("conv_layers", hp.IntInterval(1, 3))
#how big do you want the convolution kernal to be? this is 3x3, 5x5 or 7x7
HP_CONV_KERNEL_SIZE = hp.HParam("conv_kernel_size", hp.Discrete([3, 5, 7]))
#what activation function? this is relu, sigmoid and hyperbolic tangent 
HP_CONV_ACTIVATION = hp.HParam("conv_activation", hp.Discrete(["relu","sigmoid","tanh"]))
#how many convolutional kernels do you want (i.e., how many feautres should be extracted)? 
#note this is for the top layer. It will double as you go down the CNN levels. 
HP_CONV_KERNELS = hp.HParam('num_of_kernels', hp.Discrete([4,8,16,32]))
#how many dense layers do you want? this says 1 or 2
HP_DENSE_LAYERS = hp.HParam("dense_layers", hp.IntInterval(1, 2))
#what activation do you want for the dense layers?
HP_DENSE_ACTIVATION = hp.HParam("dense_activation", hp.Discrete(["relu","sigmoid","tanh"]))
#how much dropout? this changes the probability of dropout from 5% to 50% of neurons in the dense layer 
HP_DROPOUT = hp.HParam("dropout", hp.RealInterval(0.05, 0.5))
#which optimizer do you want to use? 
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adam", "adagrad","sgd","rmsprop"]))
#how many neurons do you want in the dense layer(s)? 
HP_NUM_NEURONS = hp.HParam('num_of_neurons', hp.Discrete([4,8,16,32]))
#do you want batchnorm on or off?
HP_BATCHNORM = hp.HParam('batchnorm', hp.Discrete([True,False]))
#how big of a batch do you want to try? 
HP_BATCHSIZE = hp.HParam('batch_size', hp.Discrete([32,64,128,256,512]))

## you can add additional ones, just follow the examples above and add to the list below. 

HPARAMS = [HP_CONV_LAYERS,
    HP_CONV_KERNEL_SIZE,
    HP_CONV_ACTIVATION,
    HP_CONV_KERNELS,
    HP_DENSE_LAYERS,
    HP_DROPOUT,
    HP_OPTIMIZER,
    HP_NUM_NEURONS,
    HP_DENSE_ACTIVATION,
    HP_BATCHNORM,
    HP_BATCHSIZE,
]

################################ /Hyperparameter choices ######################################

##################################### Metrics ######################################
#these are the metrics you want in your tensorboard
METRICS = [
    hp.Metric(
        "epoch_loss",
        group="validation",
        display_name="loss (val.)",
    ),
    hp.Metric(
        "epoch_loss",
        group="train",
        display_name="loss (train)",
    ),
    hp.Metric(
        "epoch_binary_accuracy",
        group="train",
        display_name="accuracy (train)",
    ),
    hp.Metric(
        "epoch_binary_accuracy",
        group="validation",
        display_name="accuracy (val.)",
    ),
    hp.Metric(
        "epoch_max_csi",
        group="validation",
        display_name="max csi (val.)",
    ),
    hp.Metric(
        "epoch_max_csi",
        group="train",
        display_name="max csi (train)",
    ),
]
##################################### /Metrics ######################################


##################################### Function Defs #######################################

def model_fn(hparams, seed,init_seed=42):
    """Create a Keras model with the given hyperparameters.
    Args:
      hparams: A dict mapping hyperparameters in `HPARAMS` to values.
      seed: A hashable object to be used as a random seed (e.g., to
        construct dropout layers in the model).
    Returns:
      A compiled Keras model.
    """
    #get random seed for weights 
    rng = random.Random(seed)

    #build inital sequential model 
    model = tf.keras.models.Sequential()
    #add input layer with defined shape 
    model.add(tf.keras.layers.Input(INPUT_SHAPE))

    # Add convolutional layers

    #how many filters is from the hparams dict. 
    conv_filters = hparams[HP_CONV_KERNELS]

    #activation is from hparam dict.     
    conv_activation = hparams[HP_CONV_ACTIVATION]

    #add CNN layers, where the number is defined by the hparam dict. 
    for _ in range(hparams[HP_CONV_LAYERS]):
        #add layer with prescribed quantities
        model.add(
            tf.keras.layers.Conv2D(
                filters=conv_filters,
                kernel_size=hparams[HP_CONV_KERNEL_SIZE],
                padding="same",
                activation=conv_activation,
            )
        )
        #add maxpooling 
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, padding="same")) 
        #double filters (you can change this if you want)
        conv_filters *= 2

    #flatten CNN layers into 1d vector to feed into ANN 
    model.add(tf.keras.layers.Flatten())

    #add dropout layer for regularization (you can delete this if you want)
    model.add(tf.keras.layers.Dropout(hparams[HP_DROPOUT], seed=init_seed))

    # Add fully connected layers, same way as CNN layers
    dense_neurons = hparams[HP_NUM_NEURONS]
    batchnormbool = hparams[HP_BATCHNORM]
    dense_activation = hparams[HP_DENSE_ACTIVATION]
    for _ in range(hparams[HP_DENSE_LAYERS]):
        #build dense layer 
        model.add(tf.keras.layers.Dense(dense_neurons, activation=dense_activation))
        #if batchnorm on, add it 
        if batchnormbool:
            model.add(tf.keras.layers.BatchNormalization())
        #if you want to add dropout here you can 
        # model.add(tf.keras.layers.Dropout(hparams[HP_DROPOUT], seed=rng.random()))

        #double neurons, not needed if you dont want it. 
        dense_neurons *= 2

    # Add the final output layer. Note, this is a classification task with binary classes, so the sigmoid is used
    # if you have OUTPUT_CLASSES > 1 and its classification, then use "softmax". If you are doing regression, do "linear"
    model.add(tf.keras.layers.Dense(OUTPUT_CLASSES, activation="sigmoid")) 

    #compile the model. We have hard coded the loss here, but you can add losses to your hyperparameter choices at the top of the script.
    #note we have added a couple metrics to show up in the tensorboard.  
    model.compile(
        loss="binary_crossentropy",
        optimizer=hparams[HP_OPTIMIZER],
        metrics=["binary_accuracy",MaxCriticalSuccessIndex()],
    )
    
    #return the built model. 
    return model

def prepare_data():
    """This is a function to load my data. Note this data is 'small' (< 2 GB). 
    If you have a bigger dataset (~16GB or >), talk to me about how you can leverage better things to do this more effciently.
    """
    # We'll use the raw downsampled images (48,48)
    ds_t = xr.open_dataset('/ourdisk/hpc/ai2es/datasets/auto_archive_notyet/tape_2copies/sub-sevir/comb/sub-sevir-train.zarr',engine='zarr')
    ds_v = xr.open_dataset('/ourdisk/hpc/ai2es/datasets/auto_archive_notyet/tape_2copies/sub-sevir/comb/sub-sevir-val.zarr',engine='zarr')

    #put data into tensorflow datasets so we can shuffle and batch in the main loop 
    ds_train = tf.data.Dataset.from_tensor_slices((ds_t.features.astype('float16').values,ds_t.label_1d_class.astype('float16').values))
    ds_val = tf.data.Dataset.from_tensor_slices((ds_v.features.astype('float16').values,ds_v.label_1d_class.astype('float16').values))

    return (ds_train, ds_val)


def run(data, base_logdir, session_id, hparams):
    """Run a training/validation session.
    Flags must have been parsed for this function to behave.
    Args:
      data: The data as loaded by `prepare_data()`.
      base_logdir: The top-level logdir to which to write summary data.
      session_id: A unique string ID for this session.
      hparams: A dict mapping hyperparameters in `HPARAMS` to values.
    """
    #clear seeds for reproducability 
    tf.keras.backend.clear_session()

    #build the model based on the hparam dict, set random seed to the current hparam session for uniqueness 
    model = model_fn(hparams=hparams, seed=session_id)
    #set the location to dump the tensorboard logs 
    logdir = os.path.join(base_logdir, session_id)

    #split apart the data 'list' that is provided by the prepare_data() method 
    ds_train,ds_val = data

    #shuffle the training data 
    ds_train = ds_train.shuffle(ds_train.cardinality().numpy(),seed=42)
    #batch the training data 
    ds_train = ds_train.batch(hparams[HP_BATCHSIZE])
    #batch the validation data, note this is not required but helps with RAM/memory things 
    ds_val = ds_val.batch(hparams[HP_BATCHSIZE])

    # have it dump tensorboard updates at the end of every epoch 
    callback = tf.keras.callbacks.TensorBoard(
        logdir,
        update_freq='epoch',
        profile_batch=0, 
    )

    #build the hparam callback 
    hparams_callback = hp.KerasCallback(logdir, hparams)

    #setup early stopping so we dont waste time training models that have converged within 5 epochs (i.e., loss did not change for 5 epochs)
    callback_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
    #here is the beef of the script. The actual training steps 
    result = model.fit(ds_train,
        epochs=flags.FLAGS.num_epochs,
        shuffle=False,
        validation_data=ds_val,
        callbacks=[callback, hparams_callback,callback_es],verbose=0) #if you want alot of output with the training annimation to the out file, change verbose=0 to verbose=1
    
    #save trained model, need to build path first. This is very hacky and specific to my file dir structure. So be advised this will be different for you. 
    split_dir = logdir.split('CNN')
    split_dir2 = split_dir[1].split('scalar_logs')
    right = split_dir2[0][:-1] + split_dir2[1]
    left = '/scratch/randychase/models/CNN'
    model.save(left + right + "model.h5")



def run_all(logdir, verbose=False):
    """Perform random search over the hyperparameter space.
    Arguments:
      logdir: The top-level directory into which to write data. This
        directory should be empty or nonexistent.
      verbose: If true, print out each run's name as it begins.
    """

    #load in your data (which is a list of (ds_train,ds_val))
    data = prepare_data()

    #set the random seed for the hyperparam search, otherwise your first hparam choice will be different everytime you run this script. 
    rng = random.Random(0)

    #this is for the tensorboard 
    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams_config(hparams=HPARAMS, metrics=METRICS)

    #this is just that we only do one training session per hyperparam set 
    sessions_per_group = 1
    num_sessions = flags.FLAGS.num_session_groups * sessions_per_group
    session_index = 0  # across all session groups
    
    #loop over all hyperparam configurations
    for group_index in range(flags.FLAGS.num_session_groups):
        #grab a random set of hyperparam choices 
        hparams = {h: h.domain.sample_uniform(rng) for h in HPARAMS}
        #store the values for the tensorboard 
        hparams_string = str(hparams)
        #this loop is useless for this setup, but lets not change it. 
        for repeat_index in range(sessions_per_group):
            session_id = str(session_index)
            #this counts what model simulation we are on 
            session_index += 1

            #if you have the verbose flag set to true, it will print stuff out to the out file. 
            if verbose:
                print(
                    "--- Running training session %d/%d"
                    % (session_index, num_sessions)
                )
                print(hparams_string)
                print("--- repeat #: %d" % (repeat_index + 1))

            #okay, time to pass the data and hparam choices to the training session. 
            run(
                data=data,
                base_logdir=logdir,
                session_id=session_id,
                hparams=hparams,
            )


#okay this will actually be the running bit for the script. 
def main(unused_argv):
    np.random.seed(0)
    logdir = flags.FLAGS.logdir
    print('removing old logs')
    shutil.rmtree(logdir, ignore_errors=True)
    print("Saving output to %s." % logdir)
    run_all(logdir=logdir, verbose=True)
    print("Done. Output saved to %s." % logdir)

##################################### /Function Defs #######################################

if __name__ == "__main__":
    app.run(main)