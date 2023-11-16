#!/usr/bin/env python
# coding: utf-8

# In[21]:


import os
import numpy as np
import pandas as pd
import math
import matplotlib.pylab as plt
import seaborn as sns
from IPython.display import display
from sklearn.linear_model import LogisticRegression


# In[22]:


# Load data
df_SIMMONS =pd.read_excel("Simmons-data-raw.xlsx")
df_SIMMONS.head(20)
  


# In[23]:


df_SIMMONS.dtypes


# In[24]:


predictor_cols = ["X1 = Spending(000)", "X2 = Loyalty Card (zero or 1)"]
target_col="Y = Coupon-Usage-Indicator"


# In[29]:


model = LogisticRegression(max_iter=1000, C=1000)
model.fit(df_SIMMONS[predictor_cols],df_SIMMONS[target_col])
beta0 = model.intercept_[0]
beta1 = model.coef_[0][0]
beta2 = model.coef_[0][1]


print('LR coefficients:')
print('BETA0 (or constant term): {}'.format(beta0))
print('BETA1 (coeff. For X1 ): {}'.format(beta1))
print('BETA2 (coeff. For X2): {}'.format(beta2))


# In[30]:


def predict_coupon_usage(X):
    model = LogisticRegression()
    model.fit(df_SIMMONS[predictor_cols],df_SIMMONS[target_col])

jack_prob = model.predict_proba([[2, 1]])[0][1]
jill_prob = model.predict_proba([[4, 0]])[0][1]
    
print('Probability of coupon usage for Jack:', jack_prob)
print('Probability of coupon usage for Jill:', jill_prob)


# In[31]:


count=0
for col in predictor_cols:
    print("ODDS Ratio for variable ", col,"= ", np.exp(lin_model.coef_[0,count]),"\n")
    count = count+1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:






# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




