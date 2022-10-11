Dataset Name: sub-SEVIR-engineered 
Author: Randy J. Chase 
Date Created: 16 May 2022
Email: randychase 'at' ou.edu 

Description:

This directory contains features engineered from the subsampled (temporally and spatially) 
version of the Storm Event Imagery Dataset (Veillette et al. NEURIPS 2020; 
https://proceedings.neurips.cc/paper/2020/hash/fa78a16157fed00d7a80515818432169-Abstract.html), 
named sub-sevir. The original dataset contained 10,000 unique storm `events`. Each event contained 48 (4 hrs)
384 km x 384 km patches of GOES-16 and NEXRAD data. The five variables contained are: 

1. Mid-tropospheric water vapor brightness temperature (ir069)
2. Clean infrared  brightness temperature (ir107)
3. NEXRAD vertically integrated liquid (vil)
4. Red channel visible reflectance (vis)
5. GLM lightning flashes 

While the original dataset was rich with information, the dataset size was too cumbersome 
for use in a tutorial/summer school format. To circumvent this issue,the sub-sevir dataset
subsampled the original SEVIR dataset to now have all the same spatial resolution 
(~8 km) and only included 60 mins (12 time steps) of each event. This successfully made 
the dataset smaller than 2 GB in total. 

In order to compare deep learning methods with traditional ML, we have gone ahead and 
extracted percentiles from each image. Specifically, we extracted `[0,1,10,25,50,75,90,99,100]`
percentiles through the np.percentile function. 

Directory Contents:
1. `lowres_features_train.csv`
    - csv file of training data split of events. 
2. `lowres_features_val.csv`
    - csv file of validation data split of events. 
3. `lowres_features_test.csv`
    - csv file of test data split of events. 
4. `README.md`
    - text document explaining the dataset. 

Explanation of variables:  

Each csv file contains 36 columns that are the `features`, 9 percentiles from each variable.
The column names have a `q_PERCENTILE_VARIABLE`. Where percentile is filled with the percentile 
values, and variable is a shorthand for the variable name. These features have been scaled to have 
mean 0 and std 1 (on the images, not the variables here). The scalers to return the data to
their original units are: 

          mean    ;   std
ir069: -37.076042 ; 11.884567
ir107: -16.015541 ; 25.805252
vil  : 0.40534842 ; 1.9639382
vis  : 0.14749236 ; 0.21128087

The equation is (x \times std) + mean.

The last two columns of each csv also contains the `label` for that sample. The label is 
either binary, for the classification task [0 no lightning, 1 lighting], or the number of 
flashes in that image [some int value > 0]. The column names for each is `label_class` and 
`label_reg` for classification and regression respectively. 

