#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from .normalizer import MinMax01Scaler,MinMax11Scaler,StdScaler
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split


# In[2]:


class ERA5Getter(Dataset):
    def __init__(self, data, input_len):
        self.data = data
        self.input_len = input_len
        
    def __getitem__(self, index):
        chunk = self.data[index]
        return chunk[:self.input_len], chunk[self.input_len:]

    def __len__(self):
        return len(self.data)


# In[3]:


class ERA5:
    def __init__(self, config,):
        self.train_data_path = str(config['data']['train_data_path'])
        self.test_data_path = str(config['data']['test_data_path'])
        self.validate_ratio = float(config['data']['validate_ratio'])
        self.validate_random_state = int(config['data']['validate_random_state'])
        self.nomalize_method = str(config['data']['nomalizer'])
        self.input_len = int(config['model']['input_len'])
        self.output_len = int(config['model']['output_len'])
        self.batch_size = int(config['model']['batch_size'])
        self.step_interval = int(config['model']['step_interval'])
        self.test_batch_size = int(config['model']['test_batch_size'])
        self.chunk_interval = int(config['model']['chunk_interval'])
    
    def sequence_chunks(self, data):
        rows_count = len(data)
        chunk_len = (self.input_len + self.output_len) * self.step_interval
        return np.array([
            range(i, i + chunk_len, self.step_interval) for i in range(0, rows_count - chunk_len+1, self.chunk_interval)
        ])        
    
    def train_loader(self,):
        
        self.train_validate_data = np.load(self.train_data_path).astype(np.float32)
        
        batch, channel, height, width = self.train_validate_data.shape
        
        if self.nomalize_method == 'std':
            self.train_nomalizer = []
            for i in range(channel):
                self.train_nomalizer.append(StdScaler(self.train_validate_data[:, i]))
        elif self.nomalize_method == 'minmax01':
            self.train_nomalizer = []
            for i in range(channel):
                self.train_nomalizer.append(MinMax01Scaler(self.train_validate_data[:, i]))
        elif self.nomalize_method == 'minmax11':
            self.train_nomalizer = []
            for i in range(channel):
                self.train_nomalizer.append(MinMax11Scaler(self.train_validate_data[:, i]))
        else:
            raise Exception("Only std, minmax01, minmax11 nomalize methods are supported")
        
        slice_data = []
        for i in range(channel):
            slice_data.append(self.train_nomalizer[i].tranform())
        self.train_validate_data = np.stack(slice_data, axis=1)
        
        self.train_validate_index = self.sequence_chunks(self.train_validate_data)
        
        self.train_index, self.validate_index = train_test_split(self.train_validate_index,test_size=self.validate_ratio, random_state=self.validate_random_state, shuffle = True)
        
        self.train_getter = ERA5Getter(self.train_index, input_len= self.input_len)
        
        return DataLoader(dataset=self.train_getter,batch_size=self.batch_size,shuffle=True, drop_last=True)
    
    def validate_loader(self, ):
        self.validate_getter = ERA5Getter(self.validate_index, input_len= self.input_len)
        return DataLoader(dataset=self.validate_getter,batch_size=self.batch_size,shuffle=True, drop_last=True)
    
    def test_loader(self, ):
        
        self.test_data = np.load(self.test_data_path).astype(np.float32)
        
        batch, channel, height, width = self.test_data.shape
        
#         print("self.test_data:", self.test_data.shape)
#         print("self.test_data[:,0]:", self.test_data[:,0].min(), self.test_data[:,0].mean(), self.test_data[:,0].max())
#         print("self.test_data[:,1]:", self.test_data[:,1].min(), self.test_data[:,1].mean(), self.test_data[:,1].max())
#         print("self.test_data[:,2]:", self.test_data[:,2].min(), self.test_data[:,2].mean(), self.test_data[:,2].max())
#         print("self.test_data[:,3]:", self.test_data[:,3].min(), self.test_data[:,3].mean(), self.test_data[:,3].max())
        
        if self.nomalize_method == 'std':
            self.test_nomalizer = []
            for i in range(channel):
                p1, p2 = self.train_nomalizer[i].pivot_values()
                self.test_nomalizer.append(StdScaler(self.test_data[:, i], p1, p2))
        elif self.nomalize_method == 'minmax01':
            self.test_nomalizer = []
            for i in range(channel):
                p1, p2 = self.train_nomalizer[i].pivot_values()
                self.test_nomalizer.append(MinMax01Scaler(self.test_data[:, i], p1, p2))
        elif self.nomalize_method == 'minmax11':
            self.test_nomalizer = []
            for i in range(channel):
                p1, p2 = self.train_nomalizer[i].pivot_values()
                self.test_nomalizer.append(MinMax11Scaler(self.test_data[:, i], p1, p2))
        else:
            raise Exception("Only std, minmax01, minmax11 nomalize methods are supported")
        
        slice_data = []
        for i in range(channel):
            slice_data.append(self.test_nomalizer[i].tranform())
        self.test_data = np.stack(slice_data, axis=1)
#         print("self.test_data:", self.test_data.shape)
#         print("self.test_data[:,0]:", self.test_data[:,0].min(), self.test_data[:,0].mean(), self.test_data[:,0].max())
#         print("self.test_data[:,1]:", self.test_data[:,1].min(), self.test_data[:,1].mean(), self.test_data[:,1].max())
#         print("self.test_data[:,2]:", self.test_data[:,2].min(), self.test_data[:,2].mean(), self.test_data[:,2].max())
#         print("self.test_data[:,3]:", self.test_data[:,3].min(), self.test_data[:,3].mean(), self.test_data[:,3].max())
        
        self.test_index = self.sequence_chunks(self.test_data)
        self.test_getter = ERA5Getter(self.test_index, input_len= self.input_len)
        return DataLoader(dataset=self.test_getter,batch_size=self.test_batch_size,shuffle=False, drop_last=True)


# In[6]:


if __name__ == '__main__':
    import configparser
    MODEL = 'gsconvlstm'
    DATASET = 'era5'
    config_file = '../{}_{}.config'.format(MODEL, DATASET)
    config = configparser.ConfigParser()
    config.read(config_file)
    era5 = ERA5(config)
    for (index,[batch_idx,data_index,target_index,]) in enumerate(era5.train_loader()):
        print(batch_idx)
    for (index,[batch_idx,data_index,target_index,]) in enumerate(era5.validate_loader()):
        print(batch_idx)
    for (index,[batch_idx,data_index,target_index,]) in enumerate(era5.test_loader()):
        print(batch_idx)


# In[ ]:




