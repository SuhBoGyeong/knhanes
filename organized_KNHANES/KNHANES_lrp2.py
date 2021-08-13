import os
import itertools

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from keras.utils import to_categorical
from keras import backend as K 
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input, Concatenate, Lambda, Reshape, BatchNormalization
from tensorflow.keras.activations import relu, softmax, sigmoid
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy

from tensorflow.keras.metrics import binary_accuracy, categorical_accuracy
from tensorflow.keras.callbacks import EarlyStopping
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ProgbarLogger

import tensorflow as tf
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score, classification_report

import innvestigate
import innvestigate.utils

import argparse
from numpy.random import seed
import random
from sklearn.model_selection import train_test_split

from keras.optimizers import Adam, RMSprop, SGD

features = ['HE_ALP','age','sex','E_VS_TY','processed_edu','HE_hCHOL','HE_tb2','sm_presnt','DM1_pt','LQ_1EQL','DI1_pr','DF2_dg','HE_WBC','HE_obe','ho_incm5','DC1_lt','DE1_pt','HE_BMI','HE_HP','LQ4_00','DE1_lt','DX_Q_hsty','DI6_dg','T_Q_DZ1','DM2_ag','DF2_lt','DI3_pt','N_VA','dr_month','DM2_pr','BD2','HE_vitD','BE3_31','DM2_pt','HE_tb7','pa_walk','BE3_12','ho_incm','LQ4_05','DM1_dg','DM2_lt','BM7','BS6_2','T_Tymp_rt','LQ4_06','DK8_dg','BD1','E_Q_RM','DI2_dg','T_Q_SNST2','T_Prsn_rt','DJ4_pt','mh_stress','DI1_ag','DM2_dg','DK4_lt','HE_LHDL_st2','DI1_2','N_CAROT','DJ2_lt','DC4_lt','DM1_lt','HE_rPLS','mh_suicide','pa_mid','LQ4_03','DI1_pt','HE_Uglu','DI6_lt','BS2_1','HE_Upro','DI2_lt','DC11_pr','DI5_pr','HE_tb6','marri_2','DI4_pr','house','E_DL_2','DI4_pt','DI2_pt','LQ4_08','DJ2_pr','DI2_2','incm','HE_DM','DI3_dg','DC1_ag','DC1_dg','HE_PTH','HE_hepaB','N_ASH','DI6_pt','BO1_2','DC3_lt','N_NIAC','DI1_dg','HE_RBC','HE_tb1','DC11_pt','DC11_dg','marri_1','DI3_lt','HE_BUN','DE2_dg','DK4_dg','HE_ALC','DE1_dg','EC_stt_1','E_DR_2','E_Q_FAM','DI2_pr','DK8_pt','T_Tymp3_rt','BD2_32','BD2_31','DI4_lt','HE_DMdg','DC1_pr','N_VITC','incm5','DK8_lt','E_DL_1','E_VS_MYO','DM3_ag','HE_Ubil','DI3_pr','DM3_dg','BE3_22','DX_Q_MP','DE2_lt','HE_Bplt','HE_Uph','DK4_pt','DJ4_lt','BE3_13','BE5_1','HE_HDL_st2','E_VS_DS','DC3_ag','DC3_pr','DK8_ag','DC5_ag','EQ5D','DM1_pr','graduat','T_VCds','HE_HB', 'BD1_11', 'processed_incm', 'processed_smoking', 'processed_drinking','processed_diabetes'] 
f_n=153
n_class=3
f_n=153
np.warnings.filterwarnings('ignore')
seed_value=0
seed(seed_value)
os.environ['PYTHONHASHSEED']=str(seed_value)
np.random.seed(seed=1)
random.seed(1)
################################################################################

### Get the arguements
parser = argparse.ArgumentParser(description='Implementation of XAI for RFA')

parser.add_argument('--nan_percent', type=str, default=10,       help='NAN_percent')
parser.add_argument('--n_teeth',     type=str, default='0',      help='Number of remaining teeth')
parser.add_argument('--beta',        type=str, default='0.9999', help='Value of beta')
parser.add_argument('--dataset_num', type=str, default='1',      help='dataset_num')
parser.add_argument('--dir_save',    type=str, default='results_lrp2',     help='Save directory')
parser.add_argument('--is_cb', action='store', default=False,
                    help='Class-balanced loss or not')

args = parser.parse_args()

NAN_PERCENT       = args.nan_percent
N_TEETH           = args.n_teeth
BETA              = float(args.beta)
DATASET_NUM       = args.dataset_num    # 1, 2, 3, 4
DIR_SAVE          = args.dir_save


print('\n#####################################################################')
print('# Arguements')
print('#####################################################################\n')
print('NAN_PERCENT       : {}'.format(NAN_PERCENT))
print('N_TEETH           : {}'.format(N_TEETH))
print('BETA              : {}'.format(BETA))
print('DATASET_NUM       : {}'.format(DATASET_NUM))
print('DIR_SAVE          : {}'.format(DIR_SAVE))


def make_model(input_shape, n_class):

    model = Sequential()
    
    model.add(Dense(512, activation=relu, input_shape=(153,)))   
    model.add(BatchNormalization())
    model.add(Dense(2048, activation=relu))
    model.add(BatchNormalization())
    model.add(Dense(16, activation=relu))
    model.add(BatchNormalization())
    model.add(Dense(n_class, activation=softmax))    
    
    #model.summary()

    return model

def removed_model(input_shape, n_class):

    model = Sequential()
    
    model.add(Dense(128, activation=relu, input_shape=(153,)))   
    model.add(Dense(64, activation=relu))
    #model.add(Dense(n_class, activation=softmax))    
    
    #model.summary()

    return model

################################################################################
# Data loading and pre-processing
################################################################################

## Split the data (train-val-test)

x_total=np.load('./imp_mix_x.npy')
y_total=np.load('./imp_mix_y.npy')
x_train, x_tmp, y_train, y_tmp=train_test_split(x_total,y_total, test_size=0.2)
x_test, x_val, y_test, y_val=train_test_split(x_tmp,y_tmp, test_size=0.5)
input_shape = x_train.shape[0]
'''
## One-hot-encode y_data
y_train = to_categorical(y_train)
y_val   = to_categorical(y_val)
y_test  = to_categorical(y_test)'''

################################################################################
# Making models and Training
################################################################################
'''
## Normalized weights based on inverse number of effective data per class.
labels, counts = np.unique(y_train, axis=0, return_counts=True)
E_n = (1.0 - np.power(BETA, counts)) / (1.0 - BETA)
alpha = np.reciprocal(E_n)
alpha = alpha / np.sum(alpha) * labels.shape[0]    # Normalization; sum of the alphas to be n_class
'''

'''
## Make a model and compile it

model = make_model(input_shape, n_class)
model.compile(
    loss=categorical_crossentropy,
    optimizer=RMSprop(lr=1e-4),
    metrics=[categorical_accuracy])

earlystopping = EarlyStopping(
    monitor='val_categorical_accuracy',
    patience=4
)

## Train the model   
hist = model.fit(
    x_train, y_train,
    epochs=100,
    validation_data=(x_train, y_train),
    steps_per_epoch=x_train.shape[0]//64,
    validation_steps=x_val.shape[0]//64,
    callbacks=[earlystopping]
)
'''
model=tf.keras.models.load_model('./models/model1.h5')
## Get the model accuracy and y_pred using test data
y_test_pred = model.predict(x_test)
ACCURACY = model.evaluate(x_test, y_test)[-1]    # model.evaluate() : [loss, accuracy]

################################################################################
# Feature selection through LRP
################################################################################

## Feature selection through LRP
print(model.layers[-1].activation)

## Train the model   

if model.layers[-1].activation is softmax:    # Drop the prediction layer if softmax is used at the last layer
    #model_inn = innvestigate.utils.model_wo_softmax(model)
    model_inn = make_model(input_shape, n_class)
    model_inn.compile(
    loss=categorical_crossentropy,
    optimizer=RMSprop(lr=1e-4),
    metrics=[categorical_accuracy])

    earlystopping = EarlyStopping(
        monitor='val_categorical_accuracy',
        patience=4
    )
else:
    print('00000000')
    model_inn = model
analyzer = innvestigate.create_analyzer("lrp.z", model_inn)    # Make an analyzer for model_inn
analysis_result = analyzer.analyze(x_train)                     # Get the relevance score for the input

## Aggregate the analysis result for feature selection
qwer = analysis_result.mean(axis=0)    # Mean of N data for each features

## Plot the aggregation result
x_axis = np.arange(qwer.shape[0])
fig1, ax1 = plt.subplots(figsize=[8, 6])
ax1.plot(x_axis, qwer, color='blue', label='Mean')    
ax1.set_title('imputed_{}_label_{}'.format(NAN_PERCENT, DATASET_NUM))
plt.legend(loc='upper left')
fig2, ax2 = plt.subplots(figsize=[8, 6])
ax2.plot(x_axis, np.sort(qwer), linestyle='None', marker='.', color='red', label='Mean_sorted')
ax2.vlines(x=x_axis[-20], ymin=np.sort(qwer)[0], ymax=np.sort(qwer)[-20], 
           color='blue', linestyles='dotted')
ax2.hlines(y=np.sort(qwer)[-20], xmin=x_axis[0], xmax=x_axis[-20],
           color='blue', linestyles='dotted')
# ax2.set_title('imputed_{}_label_{}'.format(idx_dataset, dataset))
ax2.set_title('Sorted_features', fontsize=20)
plt.legend(loc='upper left')

### Select highest and lowest top20 features, respectively
columns_data=features
columns_data = [e for e in columns_data]
df = pd.DataFrame(qwer[np.newaxis, :], columns=columns_data)

asdf = qwer.argsort()
mask_lowest = asdf[:20]
mask_highest = asdf[-20:]

highest_20 = df.iloc[:, mask_highest]
lowest_20  = df.iloc[:, mask_lowest]

highest_20 = highest_20.sort_values(by=0, axis=1, ascending=False)
lowest_20 = lowest_20.sort_values(by=0, axis=1, ascending=True)

highest_20 = highest_20.T.rename(columns={0:'Score'})
lowest_20 = lowest_20.T.rename(columns={0:'Score'})
###

################################################################################
# Save the result
################################################################################

dir_save = os.path.join(
    DIR_SAVE, 'lrp2_result')
if not os.path.exists(dir_save):
    os.makedirs(dir_save)

# fname_csv = os.path.join(dir_save, 'analysis_result.csv')
fname_fig2 = os.path.join(dir_save, 'fig_mean_sorted.png')

fname_highest20 = os.path.join(dir_save, 'highest20.csv')
fname_lowest20 = os.path.join(dir_save, 'lowest20.csv')

# np.savetxt(fname_csv, analysis_result, delimiter=',')

fig2.savefig(fname_fig2)

highest_20.round(3).to_csv(fname_highest20)
lowest_20.round(3).to_csv(fname_lowest20)


