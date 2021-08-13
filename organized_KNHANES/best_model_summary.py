import lime 
import lime.lime_tabular 
import sys,os, csv
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
from tensorflow.keras.layers import BatchNormalization, Activation
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

import pandas as pd

np.warnings.filterwarnings('ignore')
seed_value=0
seed(seed_value)
os.environ['PYTHONHASHSEED']=str(seed_value)
np.random.seed(seed=1)
random.seed(1)

opt1=Adam(lr=0.001)
opt2=RMSprop(lr=0.001)
opt3=SGD(lr=0.001)

#features = ['HE_ALP','age','sex','E_VS_TY','processed_edu','HE_hCHOL','HE_tb2','sm_presnt','DM1_pt','LQ_1EQL','DI1_pr','DF2_dg','HE_WBC','HE_obe','ho_incm5','DC1_lt','DE1_pt','HE_BMI','HE_HP','LQ4_00','DE1_lt','DX_Q_hsty','DI6_dg','T_Q_DZ1','DM2_ag','DF2_lt','DI3_pt','N_VA','dr_month','DM2_pr','BD2','HE_vitD','BE3_31','DM2_pt','HE_tb7','pa_walk','BE3_12','ho_incm','LQ4_05','DM1_dg','DM2_lt','BM7','BS6_2','T_Tymp_rt','LQ4_06','DK8_dg','BD1','E_Q_RM','DI2_dg','T_Q_SNST2','T_Prsn_rt','DJ4_pt','mh_stress','DI1_ag','DM2_dg','DK4_lt','HE_LHDL_st2','DI1_2','N_CAROT','DJ2_lt','DC4_lt','DM1_lt','HE_rPLS','mh_suicide','pa_mid','LQ4_03','DI1_pt','HE_Uglu','DI6_lt','BS2_1','HE_Upro','DI2_lt','DC11_pr','DI5_pr','HE_tb6','marri_2','DI4_pr','house','E_DL_2','DI4_pt','DI2_pt','LQ4_08','DJ2_pr','DI2_2','incm','HE_DM','DI3_dg','DC1_ag','DC1_dg','HE_PTH','HE_hepaB','N_ASH','DI6_pt','BO1_2','DC3_lt','N_NIAC','DI1_dg','HE_RBC','HE_tb1','DC11_pt','DC11_dg','marri_1','DI3_lt','HE_BUN','DE2_dg','DK4_dg','HE_ALC','DE1_dg','EC_stt_1','E_DR_2','E_Q_FAM','DI2_pr','DK8_pt','T_Tymp3_rt','BD2_32','BD2_31','DI4_lt','HE_DMdg','DC1_pr','N_VITC','incm5','DK8_lt','E_DL_1','E_VS_MYO','DM3_ag','HE_Ubil','DI3_pr','DM3_dg','BE3_22','DX_Q_MP','DE2_lt','HE_Bplt','HE_Uph','DK4_pt','DJ4_lt','BE3_13','BE5_1','HE_HDL_st2','E_VS_DS','DC3_ag','DC3_pr','DK8_ag','DC5_ag','EQ5D','DM1_pr','graduat','T_VCds','HE_HB', 'BD1_11', 'processed_incm', 'processed_smoking', 'processed_drinking','processed_diabetes'] 
features = np.load('./age_0123_column.npy', allow_pickle=True)
f_n=len(features)
n_class=3

#auto imputed datas-mix
'''
x_train=np.load('./dataset/org_x_train.npy', allow_pickle=True)
y_train=np.load('./dataset/org_y_train.npy', allow_pickle=True)
x_val=np.load('./dataset/org_x_val.npy', allow_pickle=True)
y_val=np.load('./dataset/org_y_val.npy', allow_pickle=True)
x_test=np.load('./dataset/org_x_test.npy', allow_pickle=True)
y_test=np.load('./dataset/org_y_test.npy', allow_pickle=True)'''

'''x_train=np.load('./0123_x_train.npy', allow_pickle=True)
y_train=np.load('./0123_y_train.npy', allow_pickle=True)
x_val=np.load('./0123_x_val.npy', allow_pickle=True)
y_val=np.load('./0123_y_val.npy', allow_pickle=True)
x_test=np.load('./0123_x_test.npy', allow_pickle=True)
y_test=np.load('./0123_y_test.npy', allow_pickle=True)'''

x_train=np.load('./age_0123_x_train.npy', allow_pickle=True)
y_train=np.load('./age_0123_y_train.npy', allow_pickle=True)
x_val=np.load('./age_0123_x_val.npy', allow_pickle=True)
y_val=np.load('./age_0123_y_val.npy', allow_pickle=True)
x_test=np.load('./age_0123_x_test.npy', allow_pickle=True)
y_test=np.load('./age_0123_y_test.npy', allow_pickle=True)


x_total=np.concatenate((x_train, x_val, x_test), axis=0)
y_total=np.concatenate((y_train, y_val, y_test), axis=0)

#data_number=int(x_total.shape[0])
data_number=1000
model=tf.keras.models.load_model('./model54.h5')
print('the mse value is : ', model.evaluate(x_test, y_test))

y_pred=model.predict_classes(x_total)
print(y_pred)
print(y_total)
good=0
bad=0
for i in range(len(y_total)):
    if y_pred[i]==y_total[i]:
        good+=1
    else:
        bad+=1

print(good)
print(bad)
print("accuracy: ", float((good)/(good+bad)))

j=0
preds=model.predict_proba(x_total,batch_size=None, verbose=1)
preds_label=model.predict_classes(x_total)

#print(preds.shape)
print(preds)
print(preds_label)

explainer=lime.lime_tabular.LimeTabularExplainer(x_total, feature_names=features,
                            class_names=['0','1','2'])
I=[]
for i in range(data_number):
    exp=explainer.explain_instance(x_total[i],model.predict_proba, num_features=f_n)

    #plt.show(exp.as_pyplot_figure())
    a=exp.as_list()
    I.append(a)
    print('#: ',i)
I=np.array(I)
#print(I.shape)
print(I)
#quit()
    
#mean=np.empty((f_n,1))
total_pos=np.empty(f_n)
total_neg=np.empty(f_n)
total=np.empty(f_n)
total_abs=np.empty(f_n)

#for all classes
'''for j in range(data_number): 
    for k in range(f_n):
        string_data=str(I[j][k][0])
        for feature in features:
            if str(feature) in string_data:
                if float(I[j][k][1])>0:
                    total_pos[np.where(features==feature)]+=abs(float(I[j][k][1]))
                    total[np.where(features==feature)]+=(float(I[j][k][1]))
                    total_abs[np.where(features==feature)]+=abs(float(I[j][k][1]))
                else:
                    total_neg[np.where(features==feature)]+=abs(float(I[j][k][1]))
                    total[np.where(features==feature)]+=(float(I[j][k][1]))
                    total_abs[np.where(features==feature)]+=abs(float(I[j][k][1]))
                #total[np.where(features==feature)]+=(float(I[j][k][1]))
                
            else:
                pass'''

#for particular class
for j in range(data_number):
    if int(y_total[j])==1:
        for k in range(f_n):
            string_data=str(I[j][k][0])
            for feature in features:
                if str(feature) in string_data:
                    if float(I[j][k][1])>0:
                        total_pos[np.where(features==feature)]+=abs(float(I[j][k][1]))
                        total[np.where(features==feature)]+=(float(I[j][k][1]))
                        total_abs[np.where(features==feature)]+=abs(float(I[j][k][1]))
                    else:
                        total_neg[np.where(features==feature)]+=abs(float(I[j][k][1]))
                        total[np.where(features==feature)]+=(float(I[j][k][1]))
                        total_abs[np.where(features==feature)]+=abs(float(I[j][k][1]))
                    #total[np.where(features==feature)]+=(float(I[j][k][1]))
                    
                else:
                    pass

print(total_pos)
print(total_neg)
print(total)
print(total_abs)

sorted_total_pos=np.argsort(total_pos)[::-1]
sorted_total_neg=np.argsort(total_neg)[::-1]
sorted_total=np.argsort(total)[::-1]
sorted_total_abs=np.argsort(total_abs)[::-1]
print(sorted_total_pos)
print(sorted_total_neg)
print(sorted_total)
print(sorted_total_abs)
sorted_features_pos=np.empty(f_n,dtype='str')
sorted_features_neg=np.empty(f_n,dtype='str')
sorted_features=np.empty(f_n,dtype='str')
sorted_features_abs=np.empty(f_n,dtype='str')
'''
for i in range(20):
    print(features[sorted_total_pos[i]])
    #sorted_features[i]=str(features[sorted_total[i]])
print('--------------------------------')
for i in range(20):
    print(features[sorted_total_neg[i]])
    #sorted_features[i]=str(features[sorted_total[i]])
print('--------------------------------')
for i in range(20):
    print(features[sorted_total[i]])
    #sorted_features[i]=str(features[sorted_total[i]])
print('--------------------------------')'''

f=open('./features_age.txt', 'w')
for i in range(20):
    f.write(features[sorted_total_abs[i]])
    f.write('\n')
    #print(features[sorted_total_abs[i]])
    #sorted_features[i]=str(features[sorted_total[i]])
print('--------------------------------')
f.write('\n\n')
for i in range(10):
    f.write(features[sorted_total_pos[i]])
    f.write('\n')
    #print(features[sorted_total_pos[i]])
    #sorted_features[i]=str(features[sorted_total[i]])
print('--------------------------------')
f.write('\n\n')
for i in range(10):
    f.write(features[sorted_total_neg[f_n-i-1]])
    f.write('\n')
    #print(features[sorted_total_neg[f_n-i-1]])




    


