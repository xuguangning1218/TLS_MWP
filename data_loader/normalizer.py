#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


# scale data to [0,1]
class MinMax01Scaler:
    def __init__(self, data, min_value=None, max_value=None):
        self.data = data
        if min_value == None:
            self.min_value = np.min(data)
        else:
            self.min_value = min_value
            
        if max_value == None:
            self.max_value = np.max(data)
        else:
            self.max_value = max_value
        
    def tranform(self,):
        self.data = (self.data - self.min_value)/(self.max_value - self.min_value)
        return self.data
    
    def reverse(self,):
        self.data = self.data * (self.max_value - self.min_value) + self.min_value
        return self.data
    
    def reverse(self, data):
        data = data * (self.max_value - self.min_value) + self.min_value
        return data
    
    def pivot_values(self,):
        return self.min_value, self.max_value


# In[3]:


# scale data to [-1,1]
class MinMax11Scaler:
    def __init__(self, data, min_value=None, max_value=None):
        self.data = data
        if min_value == None:
            self.min_value = np.min(data)
        else:
            self.min_value = min_value
            
        if max_value == None:
            self.max_value = np.max(data)
        else:
            self.max_value = max_value
        
    def tranform(self,):
        self.data = 2*(self.data - self.min_value)/(self.max_value - self.min_value)-1
        return self.data
    
    def reverse(self,):
        self.data = (self.data + 1) * (self.max_value - self.min_value)/2 + self.min_value
        return self.data
    
    def reverse(self, data):
        data = (data + 1) * (self.max_value - self.min_value)/2 + self.min_value
        return self.data
    
    def pivot_values(self,):
        return self.min_value, self.max_value


# In[4]:


# scale data to Z-Score
class StdScaler:
    def __init__(self, data, mean_value=None, var_value=None):
        self.data = data
        if mean_value == None:
            self.mean_value = np.mean(data)
        else:
            self.mean_value = mean_value
            
        if var_value == None:
            self.var_value = np.var(data)
        else:
            self.var_value = var_value
        
    def tranform(self,):
        self.data = (self.data - self.mean_value)/self.var_value
        return self.data
    
    def reverse(self,):
        self.data = self.data * self.var_value + self.mean_value
        return self.data
    
    def reverse(self, data):
        data = data * self.var_value + self.mean_value
        return data
    
    def pivot_values(self,):
        return self.mean_value, self.var_value


# In[5]:


if __name__ == '__main__':
    data = np.random.rand(10)
    print(data)
    print()
    scaler = MinMax01Scaler(data)
    print(scaler.tranform())
    print(scaler.pivot_values())
    print(scaler.reverse())
    print()
    scaler = MinMax11Scaler(data)
    print(scaler.tranform())
    print(scaler.pivot_values())
    print(scaler.reverse())
    print()
    scaler = StdScaler(data)
    print(scaler.tranform())
    print(scaler.pivot_values())
    print(scaler.reverse())
    print()


# In[ ]:




