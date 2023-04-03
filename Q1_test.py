# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:51:04 2022
对问题1进行测试，测试用例为10，遍历所有结果
@author: Lethe
"""

import numpy as np
import matplotlib.pyplot as plt

'''
优化目标函数
输入：区块分配 区块信息
输出：优化目标计算结果、存储占用'''
def target_Q1(blockAllo,blockInfo):
    # result = np.sum((blockInfo[:,2]+blockInfo[:,3])*blockAllo) #优化目标计算结果
    #更复杂的一种约束
    u = 0.3 #表示存储于云的成本:存储于本地节点 比例参数
    #0.05是存储指标和访问指标的比例参数,否则会导致访问指标不起作用
    result = 0.05*(np.sum(blockInfo[:,1])-(1-u)*np.sum(blockInfo[:,1]*blockAllo)) + np.sum((blockInfo[:,2]+blockInfo[:,3])*blockAllo)
    storage = np.sum((1-blockAllo)*blockInfo[:,1])/10 #该分配下的存储空间占用
    return result,storage

def blockAlloBit_add1(bit): #区块分配按位加1,倒序
    if (blockAllo[bit]+1)%2==0:
        if bit>0:
            blockAllo[bit] = 0
            blockAlloBit_add1(bit-1)
    else:
        blockAllo[bit] = 1
    

'''主测试部分'''
blockInfo = np.loadtxt('./Data/blockInfo10.csv', delimiter=',')
blockNum = blockInfo.shape[0] #需要分配的区块数量
blockAllo = np.zeros(blockNum) #区块分配矩阵


result = np.zeros(pow(2,blockNum)) #区块分配的目标函数计算结果
storage = np.zeros(pow(2,blockNum)) #区块分配的存储占用
allocationNum = pow(2,blockNum) #总共可能的分配数量
for index in range(allocationNum):
    #print(index)
    result[index], storage[index] = target_Q1(blockAllo,blockInfo)
    blockAlloBit_add1(blockNum-1)

resultAll = np.vstack([range(allocationNum),result,storage]).T #序号(可转换为分配)、优化目标、存储
resultAll = resultAll[resultAll[:,2]<3] #挑选出满足约束条件的结果

#所有结果
plt.plot(range(allocationNum), result,'.', markersize = 0.5)
plt.plot(range(allocationNum), storage,'.', markersize = 0.5)

#满足条件的结果
plt.plot(resultAll[:,0], resultAll[:,1],'.', markersize = 3)
plt.plot(resultAll[:,0], resultAll[:,2],'*', markersize = 3)

#最优的结果
optResult = np.argmin(resultAll[:,1])
plt.plot(resultAll[optResult,0], resultAll[optResult,1],'^', markersize = 10)
print(bin(int(resultAll[optResult,0])))












