#!/usr/bin/env python
# coding: utf-8

# In[402]:

import warnings
import matplotlib.pyplot as plt
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
from sklearn.metrics import *
import seaborn as sns
# In[230]:


from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support, average_precision_score, roc_curve, auc, accuracy_score
from keras.models import Model

import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
# In[231]:
def evaluate(clf,X,y,verbose=True):
    y_pred = clf.predict(X)
    acc = accuracy_score(y,y_pred)
    f1 = f1_score(y,y_pred,average='macro')
    print("test_acc = {} | test_f1 = {}".format(acc,f1))
    return acc


# In[232]:
from sklearn.model_selection import KFold, cross_val_score, train_test_split
kfolds = KFold(n_splits=3, shuffle=True, random_state=42)
def grid_search(X_train,y_train,model,param_grid,scoring=["f1_macro",'accuracy'],refit="accuracy"):
    print("Grid search to tune hyper-parameters...")
    gs = GridSearchCV(model,param_grid=param_grid,scoring=scoring,refit=refit,cv=kfolds)
    gs.fit(X_train,y_train)
    results = gs.cv_results_
    print("Mean CV ",refit,"=",gs.best_score_," achieved by configuration : ",gs.best_params_)
    best_idx = np.argwhere(results['rank_test_%s' % refit] == 1)[0,0]
    for scorer in scoring:
        for sample in ['test']:
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)][best_idx]
            print("Validation_{} = {}".format(scorer,sample_score_mean))
            #print("validation_",scorer,":",sample_score_mean)
    return gs


# In[403]:


from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


# In[404]:


df = pd.read_csv('data/full-data.csv')
n_steps = 5
samples = []
labels = []
i = 0

gap = 5
while i < len(df) - n_steps:
    # grab from i to i + 200
    if len(np.unique(df.iloc[i:i+n_steps,-1].values)) > 1:
        i = i + n_steps
        continue
    samples.append(df.iloc[i:i+n_steps,-2].values)
    #labels.append(df.iloc[i:i+n_steps,-1].values)
    labels.append(df.iloc[i,-1])
    i += gap
samples = np.concatenate(samples).reshape(-1,n_steps,1)
#labels = np.concatenate(labels).reshape(-1)
labels = np.array(labels).reshape(-1,1)


# In[410]:


label_map = np.unique(labels.reshape(-1))
label_map = {label:i for i,label in enumerate(label_map)}

id2label = np.unique(labels.reshape(-1))
id2label = ['Fridge','Macbook Air','Washing machine']


def one_hot_encoding(labels,label_map):
    one_hot_labels = np.zeros((labels.shape[0],len(label_map)))
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            label = labels[i,j]
            one_hot = np.zeros(3)
            one_hot[label_map[label]] = 1
            one_hot_labels[i] = one_hot
    return one_hot_labels


# In[415]:


def index_labelling(labels,label_map):
    index_labels = labels.copy()
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            label = labels[i,j]
            index_labels[i,j] = label_map[label]
    return index_labels


ohe_labels=one_hot_encoding(labels,label_map)




# Saving data

# In[418]:


# import pickle
# pickle.dump((x_train, x_test, y_train, y_test),open("data.pickle","wb"))


# # Modelling

# ### 1) Deep Learning approach

# In[563]:


import keras
from keras.layers import *
x_train, x_test, y_train, y_test = train_test_split(
    samples,ohe_labels, test_size=0.33, random_state=42)
import seaborn as sns
# ### CNN 1D

# In[564]:
cnn1_acc,svm_acc,cnn2_acc,rf_acc = [],[],[],[]
epochs = []
# for epoch in range(121,200):
#     for i in range(5):
#         print("--------------------------------------------")
#         print("Epoch {}".format(epoch))
#         epochs.append(epoch)
#         n_features = 1
#         # define model
#         model = keras.Sequential()
#         #model.add(BatchNormalization())
#         k = 3
#         model.add(Conv1D(filters=64, kernel_size=k, activation='relu', input_shape=(n_steps, n_features),name='conv'))
#         model.add(MaxPooling1D(pool_size=k))
#         model.add(Flatten(name='flatten'))
#         #model.add(BatchNormalization())
#         #model.add(Dense(24, activation='relu',name='dense'))
#         model.add(Dense(3,activation='softmax',name='classifier'))
#         model.compile(optimizer='adam', loss='mse',metrics=['acc'])
#
#
#
#         from keras.callbacks import EarlyStopping
#         #early_stopping = EarlyStopping(monitor='val_loss', patience=50)
#         model.fit(x_train, y_train, validation_split=0.1, epochs=epoch,batch_size=16,
#                   verbose=0)
#
#         y_pred = model.predict(x_test)
#         y_pred, y_test_label = np.argmax(y_pred,axis=-1),np.argmax(y_test,axis=-1)
#         acc = accuracy_score(y_pred,y_test_label)
#         f1 = f1_score(y_pred,y_test_label,average='macro')
#         print("test_acc = {} | test_f1 = {}".format(acc,f1))
#         cnn1_acc.append(acc)
#
#         # model.save("model.h5")
#         # print("Saved model to disk")
#         # In[27]:
#
#
#         y_pred = model.predict(x_test)
#         y_pred, y_test_label = np.argmax(y_pred,axis=-1),np.argmax(y_test,axis=-1)
#         acc = accuracy_score(y_pred,y_test_label)
#         f1 = f1_score(y_pred,y_test_label,average='macro')
#         print("test_acc = {} | test_f1 = {}".format(acc,f1))
#
#
#         # ### 2) Traditional approach
#
#         # In[307]:
#         index_labels = index_labelling(labels,label_map)
#         x_train, x_test, y_train, y_test = train_test_split(
#             samples.reshape(-1,n_steps),index_labels, test_size=0.33, random_state=42)
#
#         # ### 3) Combined
#         # Using pretrained CNN to extract features
#
#         # In[604]:
#
#         # from keras.models import load_model
#         # # load model
#         # model = load_model('model_improve.h5')
#
#         # In[606]:
#         extract = Model(model.inputs,outputs=model.get_layer('flatten').output)
#         train_features = extract.predict(x_train.reshape(-1,n_steps,1))
#         test_features = extract.predict(x_test.reshape(-1,n_steps,1))
#
#
#         # In[607]:
#
#
#         filters = model.get_layer('conv').get_weights()[0]
#         f_min, f_max = filters.min(), filters.max()
#         filters = (filters - f_min) / (f_max - f_min)
#
#         # rf = Pipeline([
#         #     ('rs', RobustScaler()),
#         #     ('rf', RandomForestClassifier(random_state=0,n_estimators=500))
#         # ])
#         # param_dist = {"rf__max_depth": [6, 7, 8, 9, 10],
#         #               "rf__min_samples_leaf": [3, 4, 5, 6, 7]
#         #               }
#         # gs_rf = grid_search(train_features, y_train, rf, param_dist)
#         # acc = evaluate(gs_rf,test_features,y_test)
#         # rf_acc.append(acc)
#
#         svm = Pipeline([
#                 ('rs',RobustScaler()),
#                 #('svm',SVC(kernel='linear',random_state=0,probability=True))
#                 ('svm',OneVsRestClassifier(LinearSVC(loss='l2', penalty='l1', dual=False)))
#             ])
#         params = {"svm__estimator__C":np.logspace(-4, 2, num=30, base=2)}
#         gs_svm = grid_search(train_features,y_train,svm,params)
#
#
#         # In[614]:
#         acc = evaluate(gs_svm,test_features,y_test)
#         svm_acc.append(acc)
#         svm_fridge, svm_macbook, svm_washing = gs_svm.best_estimator_['svm'].estimators_
#         pos_imp_feats = [0] * len(id2label)
#         neg_imp_feats = [0] * len(id2label)
#
#
#         # ### Visualize filter
#
#         # ### a) Fridge
#         clf = svm_fridge
#         full_svm_feature_imp = pd.DataFrame(zip(clf.coef_[0],list(range(64))), columns=['Value','Feature'])
#         # full_svm_feature_imp['imp'] = abs(full_svm_feature_imp['Value'])
#         full_svm_feature_imp = full_svm_feature_imp[full_svm_feature_imp['Value'] != 0]
#
#         pos_imp_feats[label_map['fridge']] = list(full_svm_feature_imp[full_svm_feature_imp['Value'] > 0]['Feature'])
#         neg_imp_feats[label_map['fridge']] = list(full_svm_feature_imp[full_svm_feature_imp['Value'] < 0]['Feature'])
#
#         # ### Macbook air
#         clf = svm_macbook
#         full_svm_feature_imp = pd.DataFrame(zip(clf.coef_[0],list(range(64))), columns=['Value','Feature'])
#         # full_svm_feature_imp['imp'] = abs(full_svm_feature_imp['Value'])
#         full_svm_feature_imp = full_svm_feature_imp[full_svm_feature_imp['Value'] != 0]
#
#         pos_imp_feats[label_map['macbook air']] = list(full_svm_feature_imp[full_svm_feature_imp['Value'] > 0]['Feature'])
#         neg_imp_feats[label_map['macbook air']] = list(full_svm_feature_imp[full_svm_feature_imp['Value'] < 0]['Feature'])
#
#         # ### Washing
#         clf = svm_washing
#         full_svm_feature_imp = pd.DataFrame(zip(clf.coef_[0],list(range(64))), columns=['Value','Feature'])
#         full_svm_feature_imp = full_svm_feature_imp[full_svm_feature_imp['Value'] != 0]
#
#         pos_imp_feats[label_map['washing machine']] = list(full_svm_feature_imp[full_svm_feature_imp['Value'] > 0]['Feature'])
#         neg_imp_feats[label_map['washing machine']] = list(full_svm_feature_imp[full_svm_feature_imp['Value'] < 0]['Feature'])
#
#
#
#         # ### 3.2) Re-train CNN
#
#         # In[624]:
#
#
#         x_train, x_test, y_train, y_test = train_test_split(
#             samples,ohe_labels, test_size=0.33, random_state=42)
#
#         # In[625]:
#
#
#         weights = np.zeros((64,3))+1e-5
#         bias = np.random.normal(size=(3))
#         # pos_imp_feats = [[13,15,16,44,63],[],[40,48]]
#         # neg_imp_feats = [[40,48],[13,15,16,63],[]]
#         for i,imp in enumerate(pos_imp_feats):
#             for w in imp:
#                 weights[w,i] = 1
#         for i,imp in enumerate(neg_imp_feats):
#             for w in imp:
#                 weights[w,i] = -1
#
#
#         # In[626]:
#         model.get_layer('classifier').set_weights((weights,bias))
#         model.compile(optimizer='adam', loss='mse',metrics=['acc'])
#
#         # In[628]:
#
#
#         #early_stopping = EarlyStopping(monitor='val_loss', patience=50)
#         history = model.fit(x_train, y_train, validation_split=0.1, epochs=epoch,batch_size=16,verbose=0)
#
#         y_pred = model.predict(x_test)
#         y_pred, y_test_label = np.argmax(y_pred,axis=-1),np.argmax(y_test,axis=-1)
#         acc = accuracy_score(y_pred,y_test_label)
#         f1 = f1_score(y_pred,y_test_label,average='macro')
#         print("test_acc = {} | test_f1 = {}".format(acc,f1))
#         cnn2_acc.append(acc)
# print(cnn1_acc)
# print(cnn2_acc)
# print(rf_acc)
# print(svm_acc)
import pickle
PIK = "acc_cnn.dat"

data = cnn1_acc,cnn2_acc,rf_acc,svm_acc
with open(PIK, "wb") as f:
    pickle.dump(data, f)
sns.lineplot(x=epochs,y=cnn1_acc)
sns.lineplot(x=epochs,y=cnn2_acc)
sns.lineplot(x=epochs,y=rf_acc)
sns.lineplot(x=epochs,y=svm_acc)
plt.legend(['Random CNN','Optimized CNN','Random forest','SVM'])
plt.xlabel('Epochs')
plt.ylabel('Test accuracy')
plt.show()

