# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:59:13 2022
产生随机输入值
@author: Lethe
"""

import numpy as np

np.random.seed(1037)

#%%
#区块init
blockNum = 1000 #区块数量

blockSize = np.random.randn(blockNum)+8 #区块大小,,正态分布,随机生成
blockFre = np.random.rand(blockNum) #区块访问频率,随机生成
#!频率完全随机有问题，因为按照真实情况，较近的区块可能频率很低
#按照最大的频率为1计算占比
blockFre = blockFre/np.max(blockFre)#标准化
blockTime = np.exp(np.arange(blockNum)-blockNum+1)

blockInfo = np.vstack((np.arange(blockNum),blockSize,blockFre,blockTime)).T
print(blockInfo)

#blockInfo按列依次为 区块序号、区块大小、区块访问频率、区块生成时间参数
np.savetxt('./Data/blockInfo'+str(blockNum)+'.csv', blockInfo, delimiter=',')

#%%
#节点init
nodeNum = 3
nodeLimit = np.random.rand(nodeNum)*500+200 #节点的存储空间
communiCost = np.zeros((nodeNum,nodeNum)) #节点通信能力矩阵
print(nodeCost)
for i in range(nodeNum): #计算任意两个节点间的通信成本
    for j in range(nodeNum):
        if i==j:
            communiCost[i,j] = 0
        else:
            communiCost[i,j] = nodeCost[i]+nodeCost[j]

nodeInfo = np.vstack([np.arange(nodeNum),nodeLimit,communiCost]).T

#nodeInfo按列依次为 节点序号、节点存储能力、节点到其他节点的通信成本
np.savetxt('./Data/nodeInfo'+str(nodeNum)+'.csv', nodeInfo, delimiter=',')

#%%
#计算区块平均大小、节点平均大小、平均通信成本
blockNum = 300
nodeNum = 10

blockInfo = np.loadtxt('./Data/blockInfo{}.csv'.format(blockNum), delimiter=',')
nodeInfo = np.loadtxt('./Data/nodeInfo{}.csv'.format(nodeNum), delimiter=',')

blockAvg = np.mean(blockInfo[:,1]) #区块平均大小
nodeAvg = np.mean(nodeInfo[:,1]) #节点平均大小
nodeSum = np.sum(nodeInfo[:,1]) #节点总存储容量
communiAvg = np.mean(nodeInfo[:,2:]) #平均通信成本

print('区块大小-平均',"%.2f" % blockAvg)
print('区块大小-范围',"%.2f" % np.min(blockInfo[:,1]),"%.2f" % np.max(blockInfo[:,1]))
print('节点大小-平均',"%.2f" % nodeAvg)
print('节点大小-范围',"%.2f" % np.min(nodeInfo[:,1]),"%.2f" % np.max(nodeInfo[:,1]))
print('节点存储区块数量-平均',int(nodeAvg/blockAvg))
print('节点存储区块数量-范围',int(np.min(nodeInfo[:,1])/blockAvg),int(np.max(nodeInfo[:,1])/blockAvg))
print('节点总存储容量',"%.2f" % nodeSum)
print('平均通信成本',"%.2f" % communiAvg)












