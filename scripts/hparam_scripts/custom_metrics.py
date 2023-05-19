import tensorflow as tf 
import tensorflow.keras.backend as K
import numpy as np

##############################################################################################################
########################################### Classification ###################################################
##############################################################################################################

class MaxCriticalSuccessIndex(tf.keras.metrics.Metric):
    """ 
    Calcualte the element-wise [e.g., pixel-wise],
    maximum Critical Success Index out of 20 thresholds.
    If update_state is called more than once, it will add give you the new max 
    csi over all calls. In other words, it accumulates the TruePositives, 
    FalsePositives and FalseNegatives. 
    
    If you want to use it on a fresh y_true,y_pred, make sure you run:

    metric.reset_state()

    then 

    metric(y_true,y_pred) #<-- this will also store the counts of TP etc.

    The shapes of y_true and y_pred should match 
    
    """ 

    def __init__(self, name="max_csi",
                 tp = tf.keras.metrics.TruePositives(thresholds=np.arange(0.05,1.05,0.05).tolist()),
                 fp = tf.keras.metrics.FalsePositives(thresholds=np.arange(0.05,1.05,0.05).tolist()),
                 fn = tf.keras.metrics.FalseNegatives(thresholds=np.arange(0.05,1.05,0.05).tolist()),
                 **kwargs):
        super(MaxCriticalSuccessIndex, self).__init__(name=name, **kwargs)

        #initialize csi value, if no data given, it will be 0 
        self.csi = self.add_weight(name="csi", initializer="zeros")

        #store defined metric functions
        self.tp = tp 
        self.fp = fp 
        self.fn = fn
        #flush functions just in case 
        self.tp.reset_state()
        self.fp.reset_state()
        self.fn.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        if (len(y_true.shape[1:]) > 2 ) and (y_true.shape[-1] == 2):
            #convert back to 1 map 
            y_true = tf.where(y_true[:,:,:,1]>0,1,0)
            #ypred[:,:,:,0] = 1 - y_pred[:,:,:,1]
            y_pred = y_pred[:,:,:,1]
            #ravel for pixelwise comparison
            y_true = tf.experimental.numpy.ravel(y_true)
            y_pred = tf.experimental.numpy.ravel(y_pred)
        #if the output is a map (batch,nx,ny,nl) ravel it
        elif (len(y_true.shape[1:]) > 2):
            y_true = tf.experimental.numpy.ravel(y_true)
            y_pred = tf.experimental.numpy.ravel(y_pred)
        

        #call vectorized stats metrics, add them to running amount of each
        self.tp.update_state(y_true,y_pred)
        self.fp.update_state(y_true,y_pred)
        self.fn.update_state(y_true,y_pred)

        #calc current max csi (so we can get updates per batch)
        self.csi_val = tf.reduce_max(self.tp.result()/(self.tp.result() + self.fn.result() + self.fp.result()))

        #assign the value to the csi 'weight'
        self.csi.assign(self.csi_val)
      
    def result(self):
        return self.csi

    def reset_state(self):
        # Reset the counts 
        self.csi.assign(0.0)
        self.tp.reset_state()
        self.fp.reset_state()
        self.fn.reset_state()

##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
############################################# Regression #####################################################
##############################################################################################################

class MeanError(tf.keras.metrics.Metric):
    """ 
    Calcualte the element-wise [e.g., pixel-wise] bias (i.e., error)
    
    For alot of meteorology problems, there is often alot of 0 pixels. This biases
    the normal metrics, and makes the values in-coherent. One way around that is to 
    calcualte the metric on only the non-zero truth pixels. 
    
    If you want to use it on a fresh y_true,y_pred, make sure you run:

    metric.reset_state()

    then 

    metric(y_true,y_pred)

    The shapes of y_true and y_pred should match!
    
    """ 

    def __init__(self, name="me",
                 **kwargs):
        super(MeanError, self).__init__(name=name, **kwargs)

        #initialize cme value, if no data given, it will be 0 
        self.me = self.add_weight(name="me", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        #ravel to 1-d tensor, makes grabing just the non-zeros easier
        y_true_flat = tf.experimental.numpy.ravel(y_true)
        y_pred_flat = tf.experimental.numpy.ravel(y_pred)

        #assign the value to the conditional mean 
        self.me.assign(tf.math.reduce_mean(tf.math.subtract(y_true_flat,y_pred_flat)))
      
    def result(self):
        return self.me

    def reset_state(self):
        # Reset the counts 
        self.me.assign(0.0)

class ConditionalMeanError(tf.keras.metrics.Metric):
    """ 
    Calcualte the element-wise [e.g., pixel-wise] conditional bias (i.e., error)
    
    For alot of meteorology problems, there is often alot of 0 pixels. This biases
    the normal metrics, and makes the values in-coherent. One way around that is to 
    calcualte the metric on only the non-zero truth pixels. 
    
    If you want to use it on a fresh y_true,y_pred, make sure you run:

    metric.reset_state()

    then 

    metric(y_true,y_pred)

    The shapes of y_true and y_pred should match!
    
    """ 

    def __init__(self, name="cme",
                 **kwargs):
        super(ConditionalMeanError, self).__init__(name=name, **kwargs)

        #initialize cme value, if no data given, it will be 0 
        self.cme = self.add_weight(name="cme", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        #ravel to 1-d tensor, makes grabing just the non-zeros easier
        y_true_flat = tf.experimental.numpy.ravel(y_true)
        y_pred_flat = tf.experimental.numpy.ravel(y_pred)

        #find non-zero flash locations 
        mask = tf.where(y_true_flat>0.0)
        #check to make sure there is at least 1 flash
        if tf.equal(tf.size(mask), 0):
            pass
        else:
            y_true_flat = tf.gather(y_true_flat,indices=mask)
            y_pred_flat = tf.gather(y_pred_flat,indices=mask)

            #assign the value to the conditional mean 
            self.cme.assign(tf.math.reduce_mean(tf.math.subtract(y_true_flat,y_pred_flat)))
      
    def result(self):
        return self.cme

    def reset_state(self):
        # Reset the counts 
        self.cme.assign(0.0)

class ConditionalMeanAbsoluteError(tf.keras.metrics.Metric):
    """ 
    Calcualte the element-wise [e.g., pixel-wise] conditional absolute mean error.
    
    For alot of meteorology problems, there is often alot of 0 pixels. This biases
    the normal metrics, and makes the values in-coherent. One way around that is to 
    calcualte the metric on only the non-zero truth pixels. 
    
    If you want to use it on a fresh y_true,y_pred, make sure you run:

    metric.reset_state()

    then 

    metric(y_true,y_pred)

    The shapes of y_true and y_pred should match!
    
    """ 

    def __init__(self, name="cmae",
                 mae = tf.keras.metrics.MeanAbsoluteError(),
                 **kwargs):
        super(ConditionalMeanAbsoluteError, self).__init__(name=name, **kwargs)

        #initialize csi value, if no data given, it will be 0 
        self.cmae = self.add_weight(name="cmae", initializer="zeros")
        #store defined metric functions
        self.mae = mae 
        #flush metric
        self.mae.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        #ravel to 1-d tensor, makes grabing just the non-zeros easier
        y_true_flat = tf.experimental.numpy.ravel(y_true)
        y_pred_flat = tf.experimental.numpy.ravel(y_pred)

        #find non-zero flash locations 
        mask = tf.where(y_true_flat>0.0)
        #check to make sure there is at least 1 flash
        if tf.equal(tf.size(mask), 0):
            pass
        else:
            y_true_flat = tf.gather(y_true_flat,indices=mask)
            y_pred_flat = tf.gather(y_pred_flat,indices=mask)

            #calc mae on that new vector 
            self.mae.update_state(y_true_flat,y_pred_flat)

        #assign the value to the conditional mean 
        self.cmae.assign(self.mae.result())
      
    def result(self):
        return self.cmae

    def reset_state(self):
        # Reset the counts 
        self.cmae.assign(0.0)
        self.mae.reset_state()
        
class ConditionalRootMeanSquaredError(tf.keras.metrics.Metric):
    """ 
    Calcualte the element-wise [e.g., pixel-wise] conditional root mean square error.
    
    For alot of meteorology problems, there is often alot of 0 pixels. This biases
    the normal metrics, and makes the values in-coherent. One way around that is to 
    calcualte the metric on only the non-zero truth pixels. 
    
    If you want to use it on a fresh y_true,y_pred, make sure you run:

    metric.reset_state()

    then 

    metric(y_true,y_pred)

    The shapes of y_true and y_pred should match!
    
    """ 

    def __init__(self, name="crmse",
                 rmse = tf.keras.metrics.RootMeanSquaredError(),
                 **kwargs):
        super(ConditionalRootMeanSquaredError, self).__init__(name=name, **kwargs)

        #initialize csi value, if no data given, it will be 0 
        self.crmse = self.add_weight(name="crmse", initializer="zeros")

        #store defined metric functions
        self.rmse = rmse 
        self.rmse.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        #ravel to 1-d tensor 
        y_true_flat = tf.experimental.numpy.ravel(y_true)
        y_pred_flat = tf.experimental.numpy.ravel(y_pred)
          
        #find non-zero flash locations 
        mask = tf.where(y_true_flat>0.0)
        #check to make sure there is at least 1 flash
        if tf.equal(tf.size(mask), 0):
            pass
        else:
            y_true_flat = tf.gather(y_true_flat,indices=mask)
            y_pred_flat = tf.gather(y_pred_flat,indices=mask)

            #calc rmse on that new vector 
            self.rmse.update_state(y_true_flat,y_pred_flat)
        
        #assign the value to the conditional mean 
        self.crmse.assign(self.rmse.result())
      
    def result(self):
        return self.crmse

    def reset_state(self):
        # Reset the counts 
        self.crmse.assign(0.0)
        self.rmse.reset_state()

class ImageRootMeanSquaredError(tf.keras.metrics.Metric):
    """ 
    Calcualte the image total root mean square error. 

    Does the predicted image produce the same number of flashes?

    This function expects a 2d image prediction. [need to add exit if not the case]
    """ 

    def __init__(self, name="irmse",
                 rmse = tf.keras.metrics.RootMeanSquaredError(),
                 **kwargs):
        super(ImageRootMeanSquaredError, self).__init__(name=name, **kwargs)

        #initialize csi value, if no data given, it will be 0 
        self.irmse = self.add_weight(name="irmse", initializer="zeros")
        #store defined metric functions
        self.rmse = rmse 
        self.rmse.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        #Get sum across image 
        y_true_count = tf.math.reduce_sum(y_true,(1,2,3))
        y_pred_count = tf.math.reduce_sum(y_pred,(1,2,3))

        self.y_true_count = y_true_count
        self.y_pred_count = y_pred_count
        #calc mae on that new vector 
        self.rmse.update_state(y_true_count,y_pred_count)
        
        #assign the value to the conditional mean 
        self.irmse.assign(self.rmse.result())
      
    def result(self):
        return self.irmse

    def reset_state(self):
        # Reset the counts 
        self.irmse.assign(0.0)
        self.rmse.reset_state()

class ParaRootMeanSquaredError(tf.keras.metrics.Metric):
    """ 
    Calcualte the element-wise [e.g., pixel-wise] conditional root mean square error.
    
    For alot of meteorology problems, there is often alot of 0 pixels. This biases
    the normal metrics, and makes the values in-coherent. One way around that is to 
    calcualte the metric on only the non-zero truth pixels. 
    
    If you want to use it on a fresh y_true,y_pred, make sure you run:

    metric.reset_state()

    then 

    metric(y_true,y_pred)

    The shapes of y_true and y_pred should match!
    
    """ 

    def __init__(self, name="prmse",
                 rmse = tf.keras.metrics.RootMeanSquaredError(),
                 **kwargs):
        super(ParaRootMeanSquaredError, self).__init__(name=name, **kwargs)

        #initialize csi value, if no data given, it will be 0 
        self.prmse = self.add_weight(name="prmse", initializer="zeros")

        #store defined metric functions
        self.rmse = rmse 
        self.rmse.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_pred = tf.cast(y_pred[:, :,0], tf.float64)
        y_true = tf.cast(y_true[:, :,0], tf.float64)
        
          
        #calc rmse on that new vector 
        self.rmse.update_state(y_true,y_pred)
        
        #assign the value to the conditional mean 
        self.prmse.assign(self.rmse.result())
      
    def result(self):
        return self.prmse

    def reset_state(self):
        # Reset the counts 
        self.prmse.assign(0.0)
        self.rmse.reset_state()
        
##############################################################################################################
##############################################################################################################
##############################################################################################################