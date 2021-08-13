import sys,os
import numpy as np 
import tensorflow as tf 
from tensorflow.keras                        import backend as K
from tensorflow.python.ops        import gen_nn_ops
from tensorflow.keras.applications.vgg16     import VGG16
from tensorflow.keras.applications.vgg19     import VGG19
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool1D, Dropout, Conv1D
from tensorflow.keras.layers import BatchNormalization, Activation, GaussianNoise
#from keras_layer_normalization import LayerNormalization
from numpy.random import seed 
import random
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
from hyperas import optim
from hyperas.distributions import choice, uniform
import random
from keras import regularizers
from sklearn.metrics import roc_auc_score, auc, roc_curve
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint

np.warnings.filterwarnings('ignore')
seed_value=0
seed(seed_value)
os.environ['PYTHONHASHSEED']=str(seed_value)
np.random.seed(seed=1)
random.seed(1)


risk_factor_list = ['HE_ALP','age','sex','E_VS_TY','processed_edu','HE_hCHOL','HE_tb2','sm_presnt','DM1_pt','LQ_1EQL','DI1_pr','DF2_dg','HE_WBC','HE_obe','ho_incm5','DC1_lt','DE1_pt','HE_BMI','HE_HP','LQ4_00','DE1_lt','DX_Q_hsty','DI6_dg','T_Q_DZ1','DM2_ag','DF2_lt','DI3_pt','N_VA','dr_month','DM2_pr','BD2','HE_vitD','BE3_31','DM2_pt','HE_tb7','pa_walk','BE3_12','ho_incm','LQ4_05','DM1_dg','DM2_lt','BM7','BS6_2','T_Tymp_rt','LQ4_06','DK8_dg','BD1','E_Q_RM','DI2_dg','T_Q_SNST2','T_Prsn_rt','DJ4_pt','mh_stress','DI1_ag','DM2_dg','DK4_lt','HE_LHDL_st2','DI1_2','N_CAROT','DJ2_lt','DC4_lt','DM1_lt','HE_rPLS','mh_suicide','pa_mid','LQ4_03','DI1_pt','HE_Uglu','DI6_lt','BS2_1','HE_Upro','DI2_lt','DC11_pr','DI5_pr','HE_tb6','marri_2','DI4_pr','house','E_DL_2','DI4_pt','DI2_pt','LQ4_08','DJ2_pr','DI2_2','incm','HE_DM','DI3_dg','DC1_ag','DC1_dg','HE_PTH','HE_hepaB','N_ASH','DI6_pt','BO1_2','DC3_lt','N_NIAC','DI1_dg','HE_RBC','HE_tb1','DC11_pt','DC11_dg','marri_1','DI3_lt','HE_BUN','DE2_dg','DK4_dg','HE_ALC','DE1_dg','EC_stt_1','E_DR_2','E_Q_FAM','DI2_pr','DK8_pt','T_Tymp3_rt','BD2_32','BD2_31','DI4_lt','HE_DMdg','DC1_pr','N_VITC','incm5','DK8_lt','E_DL_1','E_VS_MYO','DM3_ag','HE_Ubil','DI3_pr','DM3_dg','BE3_22','DX_Q_MP','DE2_lt','HE_Bplt','HE_Uph','DK4_pt','DJ4_lt','BE3_13','BE5_1','HE_HDL_st2','E_VS_DS','DC3_ag','DC3_pr','DK8_ag','DC5_ag','EQ5D','DM1_pr','graduat','T_VCds','HE_HB', 'BD1_11', 'processed_incm', 'processed_smoking', 'processed_drinking','processed_diabetes'] 
f_n=153  ##same as len(risk_factor_list)
#f_n=148
#f_n=139
'''
risk_factor_list = ['HE_ALP','age','sex','E_VS_TY','HE_hCHOL','HE_tb2','sm_presnt','DM1_pt',
'LQ_1EQL','DI1_pr','DF2_dg','HE_WBC','HE_obe','ho_incm5','DC1_lt','DE1_pt','HE_BMI','HE_HP','LQ4_00','DE1_lt',
'DX_Q_hsty','DI6_dg','T_Q_DZ1','DM2_ag','DF2_lt','DI3_pt','N_VA','dr_month','DM2_pr','BD2','HE_vitD','BE3_31',
'DM2_pt','HE_tb7','pa_walk','BE3_12','ho_incm','LQ4_05','DM1_dg','DM2_lt','BM7','BS6_2','T_Tymp_rt','LQ4_06',
'DK8_dg','BD1','E_Q_RM','DI2_dg','T_Q_SNST2','T_Prsn_rt','DJ4_pt','mh_stress','DI1_ag','DM2_dg','DK4_lt',
'HE_LHDL_st2','DI1_2','N_CAROT','DJ2_lt','DC4_lt','DM1_lt','HE_rPLS','mh_suicide','pa_mid','LQ4_03','DI1_pt',
'HE_Uglu','DI6_lt','BS2_1','HE_Upro','DI2_lt','DC11_pr','DI5_pr','HE_tb6','marri_2','DI4_pr','house','E_DL_2',
'DI4_pt','DI2_pt','LQ4_08','DJ2_pr','DI2_2','incm','HE_DM','DI3_dg','DC1_ag','DC1_dg','HE_PTH','HE_hepaB','N_ASH',
'DI6_pt','BO1_2','DC3_lt','N_NIAC','DI1_dg','HE_RBC','HE_tb1','DC11_pt','DC11_dg','marri_1','DI3_lt','HE_BUN',
'DE2_dg','DK4_dg','HE_ALC','DE1_dg','EC_stt_1','E_DR_2','E_Q_FAM','DI2_pr','DK8_pt','T_Tymp3_rt','BD2_32','BD2_31',
'DI4_lt','HE_DMdg','DC1_pr','N_VITC','incm5','DK8_lt','E_DL_1','E_VS_MYO','DM3_ag','HE_Ubil','DI3_pr','DM3_dg',
'BE3_22','DX_Q_MP','DE2_lt','HE_Bplt','HE_Uph','DK4_pt','DJ4_lt','BE3_13','BE5_1','HE_HDL_st2','E_VS_DS','DC3_ag',
'DC3_pr','DK8_ag','DC5_ag','EQ5D','DM1_pr','graduat','T_VCds','HE_HB', 'BD1_11'] 
'''
#auto imputed data - dirichlet+Gaussian
'''
x_train=np.load('./dataset/imp_x_train.npy')
x_val=np.load('./dataset/imp_x_val.npy')
x_test=np.load('./dataset/imp_x_test.npy')
y_train=np.load('./dataset/y_train.npy')
y_val=np.load('./dataset/y_val.npy')
y_test=np.load('./dataset/y_test.npy')'''

# KNN + min max scaler
'''
x_train=np.load('./dataset/KNN2_x_train.npy')
x_val=np.load('./dataset/KNN2_x_val.npy')
x_test=np.load('./dataset/KNN2_x_test.npy')
y_train=np.load('./dataset/KNN2_y_train.npy')
y_val=np.load('./dataset/KNN2_y_val.npy')
y_test=np.load('./dataset/KNN2_y_test.npy')
'''
#KNN + standard scaler
'''
x_train=np.load('./dataset/Std_x_train.npy')
y_train=np.load('./dataset/Std_y_train.npy')
x_val=np.load('./dataset/Std_x_val.npy')
y_val=np.load('./dataset/Std_y_val.npy')
x_test=np.load('./dataset/Std_x_test.npy')
y_test=np.load('./dataset/Std_y_test.npy')'''
'''
x_train=np.load('./dataset/rev1_Std_x_train.npy')
y_train=np.load('./dataset/rev1_Std_y_train.npy')
x_val=np.load('./dataset/rev1_Std_x_val.npy')
y_val=np.load('./dataset/rev1_Std_y_val.npy')
x_test=np.load('./dataset/rev1_Std_x_test.npy')
y_test=np.load('./dataset/rev1_Std_y_test.npy')'''
'''
x_train=np.load('./dataset/rev2_Std_x_train.npy')
y_train=np.load('./dataset/rev2_Std_y_train.npy')
x_val=np.load('./dataset/rev2_Std_x_val.npy')
y_val=np.load('./dataset/rev2_Std_y_val.npy')
x_test=np.load('./dataset/rev2_Std_x_test.npy')
y_test=np.load('./dataset/rev2_Std_y_test.npy')'''

#fixed columns
'''
x_train=np.load('./dataset/re_KNN3_x_train.npy')
y_train=np.load('./dataset/re_KNN3_y_train.npy')
x_val=np.load('./dataset/re_KNN3_x_val.npy')
y_val=np.load('./dataset/re_KNN3_y_val.npy')
x_test=np.load('./dataset/re_KNN3_x_test.npy')
y_test=np.load('./dataset/re_KNN3_y_test.npy')'''

#fixed columns + compounding
'''
x_train=np.load('./dataset/re2_KNN3_x_train.npy')
y_train=np.load('./dataset/re2_KNN3_y_train.npy')
x_val=np.load('./dataset/re2_KNN3_x_val.npy')
y_val=np.load('./dataset/re2_KNN3_y_val.npy')
x_test=np.load('./dataset/re2_KNN3_x_test.npy')
y_test=np.load('./dataset/re2_KNN3_y_test.npy')'''
'''
x_train=np.load('./dataset/org_x_train.npy', allow_pickle=True)
y_train=np.load('./dataset/org_y_train.npy', allow_pickle=True)
x_val=np.load('./dataset/org_x_val.npy', allow_pickle=True)
y_val=np.load('./dataset/org_y_val.npy', allow_pickle=True)
x_test=np.load('./dataset/org_x_test.npy', allow_pickle=True)
y_test=np.load('./dataset/org_y_test.npy', allow_pickle=True)

y_train=y_train.astype(int)
y_val=y_val.astype(int)
y_test=y_test.astype(int)'''

x_train=np.load('./dataset/re4_KNN3_x_train.npy')
y_train=np.load('./dataset/re4_KNN3_y_train.npy')
x_val=np.load('./dataset/re4_KNN3_x_val.npy')
y_val=np.load('./dataset/re4_KNN3_y_val.npy')
x_test=np.load('./dataset/re4_KNN3_x_test.npy')
y_test=np.load('./dataset/re4_KNN3_y_test.npy')

def data():
    return x_train, y_train, x_test, y_test

def h_model(x_train, y_train, x_test, y_test):
    f_n=153
    #f_n=139
    #f_n=148

    '''
    #auto imputed data-dr+sg
    x_train=np.load('./dataset/imp_x_train.npy')
    x_val=np.load('./dataset/imp_x_val.npy')
    x_test=np.load('./dataset/imp_x_test.npy')
    y_train=np.load('./dataset/y_train.npy')
    y_val=np.load('./dataset/y_val.npy')
    y_test=np.load('./dataset/y_test.npy')'''

    #imputed data - KNN
    '''
    x_train=np.load('./dataset/KNN2_x_train.npy')
    x_val=np.load('./dataset/KNN2_x_val.npy')
    x_test=np.load('./dataset/KNN2_x_test.npy')
    y_train=np.load('./dataset/KNN2_y_train.npy')
    y_val=np.load('./dataset/KNN2_y_val.npy')
    y_test=np.load('./dataset/KNN2_y_test.npy')'''

    #KNN + standard scaler
    '''
    x_train=np.load('./dataset/Std_x_train.npy')
    y_train=np.load('./dataset/Std_y_train.npy')
    x_val=np.load('./dataset/Std_x_val.npy')
    y_val=np.load('./dataset/Std_y_val.npy')
    x_test=np.load('./dataset/Std_x_test.npy')
    y_test=np.load('./dataset/Std_y_test.npy')'''
    '''
    x_train=np.load('./dataset/rev1_Std_x_train.npy')
    y_train=np.load('./dataset/rev1_Std_y_train.npy')
    x_val=np.load('./dataset/rev1_Std_x_val.npy')
    y_val=np.load('./dataset/rev1_Std_y_val.npy')
    x_test=np.load('./dataset/rev1_Std_x_test.npy')
    y_test=np.load('./dataset/rev1_Std_y_test.npy')'''
    '''
    x_train=np.load('./dataset/re_KNN3_x_train.npy')
    y_train=np.load('./dataset/re_KNN3_y_train.npy')
    x_val=np.load('./dataset/re_KNN3_x_val.npy')
    y_val=np.load('./dataset/re_KNN3_y_val.npy')
    x_test=np.load('./dataset/re_KNN3_x_test.npy')
    y_test=np.load('./dataset/re_KNN3_y_test.npy')'''

    '''
    x_train=np.load('./dataset/re2_KNN3_x_train.npy')
    y_train=np.load('./dataset/re2_KNN3_y_train.npy')
    x_val=np.load('./dataset/re2_KNN3_x_val.npy')
    y_val=np.load('./dataset/re2_KNN3_y_val.npy')
    x_test=np.load('./dataset/re2_KNN3_x_test.npy')
    y_test=np.load('./dataset/re2_KNN3_y_test.npy')'''
    '''
    x_train=np.load('./dataset/org_x_train.npy', allow_pickle=True)
    y_train=np.load('./dataset/org_y_train.npy', allow_pickle=True)
    x_val=np.load('./dataset/org_x_val.npy', allow_pickle=True)
    y_val=np.load('./dataset/org_y_val.npy', allow_pickle=True)
    x_test=np.load('./dataset/org_x_test.npy', allow_pickle=True)
    y_test=np.load('./dataset/org_y_test.npy', allow_pickle=True)

    y_train=y_train.astype(int)
    y_val=y_val.astype(int)
    y_test=y_test.astype(int)'''

    x_train=np.load('./dataset/re4_KNN3_x_train.npy')
    y_train=np.load('./dataset/re4_KNN3_y_train.npy')
    x_val=np.load('./dataset/re4_KNN3_x_val.npy')
    y_val=np.load('./dataset/re4_KNN3_y_val.npy')
    x_test=np.load('./dataset/re4_KNN3_x_test.npy')
    y_test=np.load('./dataset/re4_KNN3_y_test.npy')


    model=Sequential()
    #model.add(Dense({{choice([256, 512, 1024])}}, input_shape=(f_n,), kernel_regularizer='l1'))
    model.add(Dense({{choice([32,64,128,256,512,1024,2048,4096])}}, input_shape=(f_n,)))
    model.add(Activation({{choice(['relu','sigmoid','tanh'])}}))
    model.add(BatchNormalization())
    #model.add(Dropout(0.3))

    model.add(Dense({{choice([4,8,16,32,64,128,256,512,1024,2048,4096])}}))
    model.add(Activation({{choice(['relu','sigmoid','tanh'])}}))
    model.add(BatchNormalization())
    #model.add(Dropout(0.3))
    '''
    model.add(Dense({{choice([8,16,32,64,128,256,512,1024])}}))
    model.add(Activation({{choice(['relu','sigmoid','tanh'])}}))
    model.add(BatchNormalization())
    #model.add(Dropout(0.3))'''
    '''
    model.add(Dense({{choice([8,16,32,64,128,256,512,1024])}}))
    model.add(Activation({{choice(['relu','sigmoid','tanh'])}}))
    model.add(BatchNormalization())'''

    model.add(Dense(3))
    model.add(Activation('softmax'))


    es=[EarlyStopping(monitor='val_loss', patience=6),
        ModelCheckpoint(filepath='./models/best_model.h5', monitor='val_loss',
                        save_best_only=True)]

    opt1=Adam(lr=0.001)
    opt2=RMSprop(lr={{choice([0.006,0.008,0.01,0.03,0.05])}})
    opt3=SGD(lr=0.001)

    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=opt2,
                    metrics=['accuracy'])


    model.fit(x_train, y_train, batch_size=32,
                epochs=50, 
                verbose=2, validation_data=(x_val, y_val),
                callbacks=es)
    score, acc=model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    y_pred=model.predict_proba(x_test)
    y_test_dummies=pd.get_dummies(y_test,drop_first=False).values
    score=roc_auc_score(y_test_dummies,y_pred)
    print('auc: ', score)

    model=tf.keras.models.load_model('./models/best_model.h5')
    
    return {'loss': -acc, 'status':STATUS_OK, 'model':model}


best_run, best_model=optim.minimize(model=h_model, 
                                    data=data,
                                    max_evals=3,
                                    algo=tpe.suggest,
                                    trials=Trials())


best_model.save('./models/model42.h5')

model=tf.keras.models.load_model('./models/model42.h5')
model.summary()

print('the mse value is : ', model.evaluate(x_test, y_test))
print('train accuracy: ', model.evaluate(x_train, y_train))
y_pred=model.predict_proba(x_test)

#y_pred=y_pred.argmax(axis=-1)
#y_test=y_test.argmax(axis=-1)
#y_test=y_test.flatten()

y_test_dummies=pd.get_dummies(y_test,drop_first=False).values
score=roc_auc_score(y_test_dummies,y_pred)
print('auc: ', score)

#y_test_roc=np.empty((len(y_test),3))

fpr=dict()
tpr=dict()
roc_auc=dict()
for i in range(3):
    fpr[i], tpr[i], _=roc_curve(y_test_dummies[:,i], y_pred[:,i])
    roc_auc[i]=auc(fpr[i],tpr[i])

#plot of a ROC curve for a specific class
for i in range(3):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area=%0.2f)' % roc_auc[i])
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.show()





