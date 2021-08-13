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
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool1D, Dropout
from numpy.random import seed 
import random
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

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
risk_factor_list = ['HE_ALP','age','sex','E_VS_TY','processed_edu','HE_hCHOL','HE_tb2','sm_presnt','DM1_pt','LQ_1EQL','DI1_pr','DF2_dg','HE_WBC','HE_obe','ho_incm5','DC1_lt','DE1_pt','HE_BMI','HE_HP','LQ4_00','DE1_lt','DX_Q_hsty','DI6_dg','T_Q_DZ1','DM2_ag','DF2_lt','DI3_pt','N_VA','dr_month','DM2_pr','BD2','HE_vitD','BE3_31','DM2_pt','HE_tb7','pa_walk','BE3_12','ho_incm','LQ4_05','DM1_dg','DM2_lt','BM7','BS6_2','T_Tymp_rt','LQ4_06','DK8_dg','BD1','E_Q_RM','DI2_dg','T_Q_SNST2','T_Prsn_rt','DJ4_pt','mh_stress','DI1_ag','DM2_dg','DK4_lt','HE_LHDL_st2','DI1_2','N_CAROT','DJ2_lt','DC4_lt','DM1_lt','HE_rPLS','mh_suicide','pa_mid','LQ4_03','DI1_pt','HE_Uglu','DI6_lt','BS2_1','HE_Upro','DI2_lt','DC11_pr','DI5_pr','HE_tb6','marri_2','DI4_pr','house','E_DL_2','DI4_pt','DI2_pt','LQ4_08','DJ2_pr','DI2_2','incm','HE_DM','DI3_dg','DC1_ag','DC1_dg','HE_PTH','HE_hepaB','N_ASH','DI6_pt','BO1_2','DC3_lt','N_NIAC','DI1_dg','HE_RBC','HE_tb1','DC11_pt','DC11_dg','marri_1','DI3_lt','HE_BUN','DE2_dg','DK4_dg','HE_ALC','DE1_dg','EC_stt_1','E_DR_2','E_Q_FAM','DI2_pr','DK8_pt','T_Tymp3_rt','BD2_32','BD2_31','DI4_lt','HE_DMdg','DC1_pr','N_VITC','incm5','DK8_lt','E_DL_1','E_VS_MYO','DM3_ag','HE_Ubil','DI3_pr','DM3_dg','BE3_22','DX_Q_MP','DE2_lt','HE_Bplt','HE_Uph','DK4_pt','DJ4_lt','BE3_13','BE5_1','HE_HDL_st2','E_VS_DS','DC3_ag','DC3_pr','DK8_ag','DC5_ag','EQ5D','DM1_pr','graduat','T_VCds','HE_HB', 'BD1_11', 'processed_incm', 'processed_smoking', 'processed_drinking','processed_diabetes'] 


print(len(risk_factor_list))

data = np.load('./data_prep/data_wo_imputation.npy', allow_pickle=True)
# print(data.shape)   #8179,875
y = np.load('./data_prep/data_y_wo_imputation.npy') -1
features = np.load('./data_prep/data_columns_wo_imputation.npy',allow_pickle=True)
print(data)
'''
df=pd.read_csv('./data_wo_imputation1.csv')
columns = list(df.columns)
features=[]
for i in range(len(columns)-1):
    a=columns[i+1].split(' ')
    features.append(a[1])
#print(features)


#print(columns)
np.save('./features.npy', features)'''
#features=np.load('./features.npy')
'''
df=pd.DataFrame(features)
df.to_csv('./features.csv')
quit()'''
'''
y = np.expand_dims(y, axis=1)
df=pd.DataFrame(y)
df.to_csv('./data_y_wo_imputation.csv',index=False)'''



ind = np.where(features == 'edu')[0][0] 
processed_edu = np.expand_dims(list(map(lambda x: x-1 if x in [3,4] else x, list(data[:,ind]))), axis=1)
data = np.concatenate((data, processed_edu), axis=1)
features = np.append(features,'processed_edu')

ind1 = np.where(features == 'ainc')[0][0] 
ind2 = np.where(features == 'cfam')[0][0]
processed_incm = np.empty((data[:,ind1].shape[0],1))
for jj in range(data[:,ind1].shape[0]):
    if data[jj,ind1] == np.nan or data[jj,ind1] == np.nan:
        processed_incm[jj,0] = np.nan
    else:
        processed_incm[jj,0] = data[jj,ind1]/np.sqrt(data[jj,ind2])
data = np.concatenate((data, processed_incm), axis=1)
features = np.append(features,'processed_incm')

ind1 = np.where(features == 'BS1_1')[0][0] 
ind2 = np.where(features == 'BS3_1')[0][0]
processed_smoking = np.empty((data[:,ind1].shape[0],1))
for jj in range(data[:,ind1].shape[0]):
    if data[jj,ind1] in [1,3]:
        processed_smoking[jj,0] = 1
    elif data[jj,ind1] == 2 and data[jj,ind2] in [1,2]:
        processed_smoking[jj,0] = 2
    elif data[jj,ind1] == 2 and data[jj,ind2] == 3:
        processed_smoking[jj,0] = 3
data = np.concatenate((data, processed_smoking), axis=1)
features = np.append(features,'processed_smoking')

ind1 = np.where(features == 'sex')[0][0] 
ind2 = np.where(features == 'BD2_1')[0][0]
ind3 = np.where(features == 'BD1_11')[0][0] 
processed_drinking = np.empty((data[:,ind1].shape[0],1))
for jj in range(data[:,ind1].shape[0]):
    if data[jj,ind1] == 1 and data[jj,ind2] in [4,5] and data[jj,ind3] in [5,6]:
        processed_drinking[jj,0] = 1
    elif data[jj,ind1] == 2 and data[jj,ind2] in [3,4,5] and data[jj,ind3] in [5,6]:
        processed_drinking[jj,0] = 1
    else:
        processed_drinking[jj,0] = 0
data = np.concatenate((data, processed_drinking), axis=1)
features = np.append(features,'processed_drinking')

ind1 = np.where(features == 'HE_glu')[0][0] 
ind2 = np.where(features == 'DE1_31')[0][0]
ind3 = np.where(features == 'DE1_32')[0][0] 
ind4 = np.where(features == 'DE1_dg')[0][0] 
processed_diabetes = np.empty((data[:,ind1].shape[0],1))
for jj in range(data[:,ind1].shape[0]):
    if data[jj,ind1] >= 126 or data[jj,ind2] == 1 or data[jj,ind3] == 1 or data[jj,ind4] == 1:
        processed_diabetes[jj,0] = 1
    else:
        processed_diabetes[jj,0] = 0
data = np.concatenate((data, processed_diabetes), axis=1)
features = np.append(features,'processed_diabetes')
'''
print(data.shape)
print(features)
df=pd.DataFrame(data)
df.to_csv('./re2_data.csv')
df=pd.DataFrame(features)
df.to_csv('./re2_features.csv')'''
#no ID in data!!!!! but ID in features ==>data=(8417,3569) features=(3570,1)

for ii, risk_factor in enumerate(risk_factor_list):     
    #ind = np.where(features == risk_factor)[0]-1
    ind = np.where(features == risk_factor)[0]
    print(ind)
    
    if  ii == 0:
        data2 = data[:,ind]
    else:
        data2 = np.concatenate((data2, data[:,ind]),axis=1)


data = data2
features=np.array(risk_factor_list)
'''
data_tmp=np.empty((data.shape[0],len(risk_factor_list)))

for i in range(len(risk_factor_list)):
    for j in range(len(features)):
        if str(features[j])==risk_factor_list[i]:
            ind=j-1
            break
    #print(ind)
    data_tmp[:,i]=data[:,ind]
    #print(data[:,ind])


'''
#data_tmp=data
#df=pd.DataFrame(data)
#df.to_csv('./re_data_tmp.csv',index=False)
#risk_features = np.array(risk_factor_list)
#print(features)
#print(data[0,:])
#quit()




#print(data.shape)
#print(risk_features.shape)


train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

data_merge = np.concatenate((data,y.reshape((y.shape[0],1))), axis=1)
np.random.shuffle(data_merge)

x_data = data_merge[:,:-1]
y = data_merge[:,-1]

#x_train, x_test, y_train, y_test=train_test_split(x_data, y, test_size=0.2, stratify=y, random_state=0)
#x_test, x_val, y_test, y_val=train_test_split(x_test, y_test, test_size=0.5, stratify=y_test, random_state=0)
'''
r=0
print(y_train)
for i in range(len(y_train)):
    if y_train[i]==2:
        r+=1
print(r/len(y_train))
r1=0
for i in range(len(y_val)):
    if y_val[i]==2:
        r1+=1
print(r1/len(y_val))
r2=0
for i in range(len(y_test)):
    if y_test[i]==2:
        r2+=1
print(r2/len(y_test))'''


train_val = int(x_data.shape[0] * train_ratio)
val_test = int(x_data.shape[0] * (train_ratio + val_ratio))

x_train = x_data[:train_val, ...]
y_train = y[:train_val, ...]
x_val = x_data[train_val:val_test, ...]
y_val = y[train_val:val_test, ...]
x_test = x_data[val_test:, ...] 
y_test = y[val_test:, ...]

#np.save('data_x.npy',x_data)
#np.save('data_y.npy',y)

##########Imputation##############
print('imputation start')
imputer = KNNImputer(n_neighbors=int(data.shape[0]/10), weights = 'uniform')
x_train=imputer.fit_transform(x_train)
x_val=imputer.transform(x_val)
x_test=imputer.transform(x_test)
#data=imputer.fit_transform(data)

#min max scaler

Scaler = MinMaxScaler(feature_range=(0,1))
x_train = Scaler.fit_transform(x_train)
x_val = Scaler.transform(x_val)
x_test = Scaler.transform(x_test)


#standard sclaer
'''
Scaler = StandardScaler()
x_train = Scaler.fit_transform(x_train)
x_val = Scaler.transform(x_val)
x_test = Scaler.transform(x_test)

x_train = np.clip(x_train, -5, 5)
x_val = np.clip(x_val, -5, 5)
x_test = np.clip(x_test, -5, 5)'''
'''
df_train=pd.DataFrame(x_train)
df_train.to_csv('./re2_KNN3_x_train.csv')
df_test=pd.DataFrame(x_test)
df_test.to_csv('./re2_KNN3_x_test.csv')
df_val=pd.DataFrame(x_val)
df_val.to_csv('./re2_KNN3_x_val.csv')'''

np.save('./dataset/org1_x_train.npy',x_train)
np.save('./dataset/org1_y_train.npy',y_train)
np.save('./dataset/org1_x_val.npy', x_val)
np.save('./dataset/org1_y_val.npy', y_val)
np.save('./dataset/org1_x_test.npy', x_test)
np.save('./dataset/org1_y_test.npy', y_test)

'''
x_data=np.concatenate((x_train,x_val,x_test), axis=0)

df=pd.DataFrame(x_data)
df.to_csv('data.csv', index=False)'''