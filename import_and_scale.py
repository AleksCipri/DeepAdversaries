import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy import arcsinh as arcsinh
from torch.utils.data import Dataset, TensorDataset, DataLoader

def array_to_tensor(name):  
  data_list = list(np.load(str(name)))
  return(torch.Tensor(data_list))

def plot_filters(t):

  plt.hist(t[:, 0].numpy().ravel(), bins=30, color = 'blue', alpha = 0.7, density=True)
  plt.hist(t[:, 1].numpy().ravel(), bins=30, color = 'red', alpha = 0.7, density=True)
  plt.hist(t[:, 2].numpy().ravel(), bins=30, color = 'green', alpha = 0.7, density=True)

  plt.xlabel("pixel values")
  plt.ylabel("relative frequency")
  plt.title("distribution of pixels")

  print('Min: %.3f, Max: %.3f' % (t[:, 0].min(), t[:, 0].max()))
  print('Min: %.3f, Max: %.3f' % (t[:, 1].min(), t[:, 1].max()))
  print('Min: %.3f, Max: %.3f' % (t[:, 2].min(), t[:, 2].max()))

def mean_std(t):
    mean1 = t[:,0].mean().item()
    mean2 = t[:,1].mean().item()
    mean3 = t[:,2].mean().item()
    mean = [mean1,mean2,mean3]

    std1 = t[:,0].std().item()
    std2 = t[:,1].std().item()
    std3 = t[:,2].std().item()
    std = [std1, std2, std3]

    return mean, std

def update_sinh(t):
  #first clip outliers based on global values
  global_min = np.percentile(t, 0.1)
  global_max = np.percentile(t, 99.9)
  #global_max = np.percentile(t, 100)
  print(global_max)

  for i in range(0, 3):
    #g, r, i
    c = .85/global_max #gets you close to arcsinh(max_x) = 1, arcsinh(min_x) = 0
    t[:,i] = np.clip(t[:,i], global_min, global_max)
    t[:,i] = arcsinh(c*t[:, i])
    t[:,i] = (t[:,i] + 1.0) / 2.0