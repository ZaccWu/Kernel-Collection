# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# %%
f1=open("E:/Tinky/大学课件及作业/6 自学课/6-3.Kaggle竞赛/C6 Santander Customer Transaction Prediction/train.csv")
f2=open("E:/Tinky/大学课件及作业/6 自学课/6-3.Kaggle竞赛/C6 Santander Customer Transaction Prediction/test.csv")
train=pd.read_csv(f1)
test=pd.read_csv(f2)

# %%
train.head()

# %%
#check the information
print(train.info())
print(train.shape)
print(train.dtypes)


#%%
#sumup
train.describe()


#%%
train.isnull().sum().sort_values()

#%%
train.nunique().sort_values()

#%%
# here interesting columns means the frequency of a value in a column exceeds 100.
fea_cols=train.columns
interesting_cols=[]
for col in fea_cols:
    if train[col].value_counts().iloc[0]>100:
        print(col,train[col].value_counts().iloc[:3])
        interesting_cols.append(col)

#%%
# check whether we have increasing or decreasing columns
def increasing(vals):
    cnt=0
    len_=len(vals)
    for i in range(len_-1):
        if vals[i+1]>vals[i]:
            cnt+=1
    return cnt

fea_cols=[col for col in train.columns if 'var' in col]
for col in fea_cols:
    cnt=increasing(train[col].values)
    if cnt/train.shape[0]>=0.55:
        print(col,cnt,cnt/train.shape[0])

#%%
train['target'].value_counts()

#%%
test.head()

#%%
test.info()

#%%
test.isnull().sum().sort_values()

#%%
test.nunique().sort_values()

#%%
# now we need to use Adversial training method to jugde whether the training set and the testing set have the same distribution
# If AUC is lower than 0.6, we can conclude that the distributions are balanced
train['label']=0
test['label']=1
trte=pd.concat([train,test],axis=0,ignore_index=True)

#%%
import lightgbm as lgb
import time
from sklearn.model_selection import KFold,StratifiedKFold

#%%
task=['A','B','C','B','A','A','D']
d=collections.Counter(task)
d

