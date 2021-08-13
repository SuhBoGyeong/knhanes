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
from tensorflow.keras.layers import BatchNormalization, Activation, GaussianNoise, AveragePooling2D
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from numpy import *
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, accuracy_score, multilabel_confusion_matrix


df=pd.read_csv('./raw/2011/hn11_dxa.csv')
age=df['age']
#df['DX_OST'].fillna(0)
DX_OST=df['DX_OST']


#print(df.loc[df['DX_OST']==' '])
df.loc[df['DX_OST']==' ']=0
DX_OST=df['DX_OST']
df.loc[df['DX_Q_MP']==' ']=0
DX_Q_MP=df['DX_Q_MP']


#print(DX_OST)
idx1=0
idx2=0

for i in range(len(age)):
    #print(df['age'][i])
    if int(DX_OST[i])==1 or 2 or 3:
        #print('-------')
        #print(age[i])
        if 0<int(age[i])<50:
            print(age[i], i, DX_Q_MP[i])
            idx1+=1

            if int(DX_Q_MP[i])!=1:
                print(DX_Q_MP[i])
                idx2+=1
            
print('idx: ', idx1, idx2)