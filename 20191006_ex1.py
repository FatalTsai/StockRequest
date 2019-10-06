
# coding: utf-8

# In[135]:


from sklearn import preprocessing 
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[136]:


train = pd.read_csv("Titanic/train.csv")
test = pd.read_csv("Titanic/test.csv")
submit = pd.read_csv('Titanic/gender_submission.csv')


# In[137]:


train = pd.read_csv("http://ez2tv.myds.me/NTHU/train.csv")
test = pd.read_csv("http://ez2tv.myds.me/NTHU/test.csv")
submit = pd.read_csv('http://ez2tv.myds.me/NTHU/gender_submission.csv')


# In[138]:


train.head(5) #取前幾個資料


# In[139]:


train.info() #看資料的主要 型態
test.info()


# In[140]:


train.describe()
#計算出各個欄位的 
#統計數值 e.g. avg. max min 50%


# In[141]:


test.describe()


# In[142]:


data = train.append(test) #將test 加價再train後面
data#會出現index相同 重複的問題


# In[143]:


data.reset_index(inplace=True, drop=True)
#重整 index直


# In[144]:


type(data)


# In[145]:


sns.countplot(data['Pclass'], hue=data['Survived'])


# In[146]:


g = sns.FacetGrid(data, col='Survived')
g.map(sns.distplot, 'Age', kde=True)


# In[147]:


x = sns.FacetGrid(data, col='Pclass')
x.map(sns.distplot, 'Age', kde=True)


# In[148]:


x = sns.FacetGrid(data, col='Survived')
x.map(sns.distplot, 'Fare', kde=True)


# In[149]:


x = sns.FacetGrid(data, col='Survived')
x.map(sns.distplot, 'Parch', kde=True)


# In[150]:


sns.countplot(data['Parch'], hue=data['Survived'])


# In[151]:


data['Family_Size']= data['Parch']+data['SibSp']


# In[152]:


sns.countplot(data['Family_Size'], hue=data['Survived'])


# In[153]:


data.head()['Name']


# In[154]:


data['Title1']=data['Name'].str.split(',',expand=True)[1]
data['Title1']


# In[155]:


data['Title1'] = data['Title1'].str.split(".", expand=True)[0]
data['Title1']


# In[156]:


data['Title1'].unique()


# In[157]:


data.groupby('Title1')['Age'].mean()


# In[158]:


pd.crosstab(data['Title1'],data['Sex']).T.style.background_gradient(cmap='summer_r')
## crosstab == Confusion Matrix


# In[159]:


pd.crosstab(data['Title1'],data['Survived']).T.style.background_gradient(cmap='summer_r')


# In[160]:


data.groupby(['Title1'])['Age'].mean()


# In[161]:


data.groupby(['Title1','Pclass'])['Age'].mean()


# In[162]:


data['Name'][759]


# In[163]:


data["Title1"].unique()


# In[168]:


data['Title2'] = data['Title1'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],['Miss','Mrs','Miss','Mr','Mr','Mrs','Mrs','Mr','Mr','Mr','Mr','Mr','Mr','Mrs'])


# In[170]:


print(data["Title2"].unique())


# In[169]:




