import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# Importing the training set
dataset = data = pd.read_csv('./all_stocks_5yr.csv')
dataset.head()

dataset_cl = dataset[dataset['Name']=='SWKS'].close.values
print(dataset_cl)

# feature scaling
sc = MinMaxScaler(feature_range = (0, 1))
# scale the data
dataset_cl = dataset_cl.reshape(dataset_cl.shape[0], 1)
dataset_cl = sc.fit_transform(dataset_cl)

print(dataset_cl)
