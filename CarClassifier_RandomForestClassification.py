
# coding: utf-8

# In[1]:


#Libraries
import pandas as pd
from sklearn import metrics ,model_selection,preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


# In[2]:


#Download the Dataset from UCI URL Directly and adding  the column names
dataset=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",names=['buying','maint','doors','persons','lug_boot','safety','class'])


# In[3]:


dataset.info()


# In[4]:


#Identify the Target Variable
dataset['class'],class_names=pd.factorize(dataset['class'])


# In[5]:


#Printing the Class value and corresponding Integer Value
print("printing the class names",class_names)
print("Printing the uniquedataset",dataset['class'].unique())


# In[6]:


#Converting each of the class varibale to its corresponding integer Value
dataset['buying'],_=pd.factorize(dataset['buying'])
dataset['maint'],_=pd.factorize(dataset['maint'])
dataset['doors'],_=pd.factorize(dataset['doors'])
dataset['persons'],_=pd.factorize(dataset['persons'])
dataset['lug_boot'],_=pd.factorize(dataset['lug_boot'])
dataset['safety'],_=pd.factorize(dataset['safety'])


# In[7]:


dataset.head()


# In[8]:


#Select feature and target variable
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]


# In[10]:


X.head(5)


# In[11]:


y.head(5)


# In[12]:


#Train Test Split
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


#Model Fitting
random_model=RandomForestClassifier(random_state=0,n_estimators=10, criterion="gini")
random_model.fit(X_train,y_train)


# In[14]:


#Model Accuracy
y_pred=random_model.predict(X_test)
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy prediction of random Forest is ",accuracy)

