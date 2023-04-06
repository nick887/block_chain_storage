# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:03:37 2023

@author: Lethe
"""
import numpy as np
import matplotlib.pyplot as plt


#%%
fileName = 'Genetic_B300_N30_BU3_L0.65_AN10_E100.npz'
data = np.load('./Result/Q2/'+fileName)['arr_0'] #总目标 通信成本 存储平衡度

targetV = np.max(data[:,0])-np.min(data[:,0])
print(targetV)
communicationV = np.max(data[:,1])-np.min(data[:,1])
print(communicationV)
nodeStorageV = np.max(data[:,2])-np.min(data[:,2])
print(nodeStorageV)

#%%

npzfile = np.load('./Result/Q2/Genetic_B100_N30_BU4_L0.65_AN10_E100_Storage.npz')

paraName = ['storageVar','storageProVar','nodeGreStorLimSum',
            'nodeGreStorLimSquaSum','nodeGreStorLimNum']

for index in range(len(paraName)):
    print(paraName[index])
    data = npzfile[paraName[index]]
    print(max(data)-min(data))
    fig = plt.figure(figsize=(6,4), dpi= 300)
    plt.plot(range(len(data)), data,'-',label=paraName[index])
    plt.legend()

#nodeGreStorLimNumLayer
data = npzfile['nodeGreStorLimNumLayer']
fig = plt.figure(figsize=(6,4), dpi= 300)
for index in range(6,9):
    plt.plot(range(len(data)), data[:,index],'-',label=index)
plt.legend()






