# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 20:07:02 2022
对问题2进行测试，测试区块数量为10，节点数量为5，遍历所有结果
@author: Lethe
"""
import numpy as np
import matplotlib.pyplot as plt

def target_Q2(blockAllo,costAll):
    nodeBlockCost = np.zeros((5,10)) #节点访问区块的代价,行代表节点,列代表区块
    
    #temp = (blockInfo[:,1]*blockInfo[:,2]).reshape((10,1)) * nodeInfo[0,2:7].reshape((1,5))
    #所有节点需要的花销shape=(blockNum,nodeNum,nodeNum)
    for node in range(nodeNum):
        for block in range(blockNum):
            nodeBlockCost[node,block] = np.min(costAll[block,blockAllo[:,block]==1,node]) #选出满足条件的

    return nodeBlockCost

#一次只选择一个节点存储区块
#从左到右按照位次递进 
def choice_node(bit): #bit代表要操作的区块序号
    if bit>=0:
        if blockLoc[bit] == nodeNum-1:
            blockLoc[bit] = 0
            choice_node(bit-1)
        else:
            blockLoc[bit] += 1


if __name__=="__main__":
    blockBackups = 1 #区块备份数量 至少为1
    storageLimit = 0.6 #优化后系统总存储空间占比
    #区块信息
    blockInfo = np.loadtxt('./Data/blockInfo8.csv', delimiter=',')
    blockNum = blockInfo.shape[0] #区块数量
    #节点信息
    nodeInfo = np.loadtxt('./Data/nodeInfo5.csv', delimiter=',')
    nodeNum = nodeInfo.shape[0] #节点数量
    
    constraint = np.full(3, False, dtype=bool)  #三个约束的结果 默认全False 
    
    #所有节点需要的花销shape=(blockNum,nodeNum,nodeNum)
    costAll = (blockInfo[:,1]*blockInfo[:,2]).reshape((blockNum,1,1)) * nodeInfo[:,2:7].reshape((1,nodeNum,nodeNum))
    
    #blockAllo = np.load('./data/tempAllo.npy')
    blockLoc = np.zeros(blockNum).astype(int)
    resultAll = np.zeros((pow(nodeNum,blockNum),4)) #保存所有目标计算结果的数组 分别为 访问目标 存储占用目标 访问+存储占用
    resultAll[:,0] = range(resultAll.shape[0]) #第一列为序号
    
    #按照一个区块只分给一个节点来遍历
    for index in range(pow(nodeNum,blockNum)): 
        blockAllo = np.zeros((nodeNum,blockNum)).astype(int) #区块分配结果,一行代表一个节点(全0不满足条件且会出错)
        for i in range(blockNum):
           blockAllo[blockLoc[i],i] = 1
        
        
        #约束条件1 每个节点的空间限制
        constraint[0] = np.all(np.dot(blockAllo,blockInfo[:,1]) <= nodeInfo[:,1]) #为True即为节点均满足要求
        #约束条件2 每个区块至少有blockBackups备份
        constraint[1] = np.all(blockAllo.sum(axis=0) >= blockBackups) #为True即为满足备份数量要求
        #约束条件3 系统总的存储空间占用率不超过限定值
        constraint[2] = (np.sum(blockAllo.sum(axis=0)*blockInfo[:,1])/np.sum(nodeInfo[:,1])<=storageLimit) #为True即为满足约束
        
        if np.all(constraint):
            resultAll[index,1] = np.sum(target_Q2(blockAllo,costAll))
            nodeProportion = np.dot(blockAllo,blockInfo[:,1])/nodeInfo[:,1] #节点存储空间占总空间的比例
            nodeProportion = (np.exp(5*nodeProportion)-1)/(np.exp(5)-1)*100 #按照指定函数对空间占用率进行处理
            resultAll[index,2] = np.sum(nodeProportion)
            resultAll[index,3] = resultAll[index,1] + resultAll[index,2]
            
        choice_node(blockNum-1)  
            
    #序号(可转换为分配) 访问目标 存储占用目标 访问+存储占用       
    #resultAll = np.vstack([range(resultAll.shape[0]),resultAll]).T 
    resultAll = resultAll[resultAll[:,3]!=0] #把为0的值踢出去(不符合约束的分配)
    #画出所有结果点
    plt.plot(resultAll[:,0], resultAll[:,3],'.', markersize = 0.5)
    #最优值
    optResult = np.argmin(resultAll[:,3])
    plt.plot(resultAll[optResult,0], resultAll[optResult,3],'^', markersize = 10)
    print(resultAll[optResult,:])
        
#任意进制转换工具 https://tool.ip138.com/hexconvert/     
        
    
    
    
    

