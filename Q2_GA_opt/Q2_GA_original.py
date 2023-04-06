# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:51:59 2022
Q2
@author: Lethe
"""

import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(1037)

'''
随机生成区块分配方案 同时判断生成的方案是否满足基本要求
nodeNum blockNum 为全局变量
完全随机找到合适的值比较慢 可优化
'''
def generate_blockAllo(alloNum=10): #默认生成100种分配
    blockAlloGen = np.zeros((alloNum,nodeNum,blockNum)).astype(bool)
    for i in range(alloNum):
        blockAllo = np.random.randint(2, size=(nodeNum,blockNum)).astype(bool)
        while(constraint(blockAllo)==False):
            blockAllo = np.random.randint(2, size=(nodeNum,blockNum)).astype(bool)
        blockAlloGen[i,:,:] = blockAllo

    return blockAlloGen

'''随机生成区块 按照备份数量生成'''
def generate_blockAllo_backups(alloNum=10):
    blockAlloGen = np.zeros((alloNum,nodeNum,blockNum)).astype(bool)
    for i in range(alloNum):
        blockAllo = np.zeros((nodeNum,blockNum)).astype(bool)
        while(constraint(blockAllo)==False):
            blockAllo = np.zeros((nodeNum,blockNum)).astype(bool)
            for block in range(blockNum):
                nodeChoice = random.sample(range(nodeNum), blockBackups)
                for index in nodeChoice:
                    blockAllo[index,block] = True
        blockAlloGen[i,:,:] = blockAllo
        
    return blockAlloGen

'''
约束条件判断
输入 区块分配方案 blockAllo
输出 True False
其中 blockInfo blockBackups nodeInfo storageLimit 为全局变量
'''
def constraint(blockAllo):
    constraint = np.full(3, False, dtype=bool)  #三个约束的结果 默认全False
    #约束条件1 每个节点的空间限制
    constraint[0] = np.all(np.dot(blockAllo,blockInfo[:,1]) <= nodeInfo[:,1]) #为True即为节点均满足要求
    #约束条件2 每个区块至少有blockBackups备份
    constraint[1] = np.all(blockAllo.sum(axis=0) >= blockBackups) #为True即为满足备份数量要求
    #约束条件3 系统总的存储空间占用率不超过限定值
    constraint[2] = (np.sum(blockAllo.sum(axis=0)*blockInfo[:,1])/np.sum(nodeInfo[:,1])<=storageLimit) #为True即为满足约束

    return np.all(constraint)

'''
计算目标函数
全局变量 costAll
'''
def objective(blockAllo):
    nodeBlockCost = np.zeros((nodeNum,blockNum)) #节点访问区块的代价,行代表节点,列代表区块
    for node in range(nodeNum):
        for block in range(blockNum):
            nodeBlockCost[node,block] = np.min(costAll[block,blockAllo[:,block]==1,node]) #选出通信成本最小的节点所需成本
    nodeProportion = np.dot(blockAllo,blockInfo[:,1])/nodeInfo[:,1] #节点存储空间占总空间的比例
    nodeProportion = (np.exp(5*nodeProportion)-1)/(np.exp(5)-1)*10 #按照指定函数对空间占用率进行处理
    
    #分别为 通信成本 存储平衡度 总目标
    #计算总目标时乘以 1/区块数量
    result = np.array([np.sum(nodeBlockCost),np.sum(nodeProportion),(np.sum(nodeBlockCost)+np.sum(nodeProportion))/nodeNum])
    return result

'''计算个体的适应度'''
def cal_fitness(blockAlloGen):
    alloNum = blockAlloGen.shape[0]
    #行代表一种分配/个体 列依次为 适应度 通信成本 存储平衡度 总目标
    alloFit = np.zeros((alloNum,4)) 
    for index,blockAllo in enumerate(blockAlloGen): #遍历
        alloFit[index,1:] = objective(blockAllo)
    
    alloFit[:,0] = -(alloFit[:,3]-np.max(alloFit[:,3]))+1e-3 #计算适应度 适应度为正数
    return alloFit

'''选择一定数量的个体'''
def select(blockAlloGen, alloFit, selectSize=1):
    idx = np.random.choice(np.arange(alloFit.shape[0]), size=selectSize, replace=False, #replace代表是否能重复抽取
                           p=(alloFit[:,0])/(alloFit[:,0].sum()) )
    return blockAlloGen[idx,:,:],alloFit[idx,:]

'''交叉'''
def crossover(blockAlloGen, CROSSOVER_RATE=0.8):
    blockAlloNew = []
    alloSize = blockAlloGen.shape[0]
    for alloFa in blockAlloGen:		#遍历种群中的每一个个体，将该个体作为父亲
        blockAlloNew.append(alloFa)
        alloChild = alloFa.copy()		#孩子先得到父亲的全部基因
        if np.random.rand() < CROSSOVER_RATE:			#以一定的概率发生交叉
            alloMa = blockAlloGen[np.random.randint(alloSize),:,:]	#再种群中选择另一个个体，并将该个体作为母亲
            crossPoints = np.random.randint(low=0, high=blockNum, size=nodeNum)	#随机产生交叉的点
            for i in range(nodeNum): #按照行交叉
                alloChild[i,crossPoints[i]:] = alloMa[i,crossPoints[i]:] #孩子得到位于交叉点后的母亲的基因 
        
            if constraint(alloChild)==True:
                print('产生子代')
                blockAlloNew.append(alloChild) #若交叉后的个体不满足约束条件 则不进行交叉
            
        #alloChild = mutation(alloChild)	#每个后代有一定的机率发生变异
        

        
        #blockAlloNew.append(alloChild)

    return np.array(blockAlloNew)

'''变异'''
def mutation(alloChild, MUTATION_RATE=0.003):
    if np.random.rand() < MUTATION_RATE: 				#以MUTATION_RATE的概率进行变异
        mutate_block = np.random.randint(0, blockNum)	#随机产生一个实数 代表要变异基因的位置/列
        mutate_node = np.random.randint(0, nodeNum)	#随机产生一个实数 代表要变异基因的位置/行
        alloChild[mutate_node, mutate_block] = alloChild[mutate_node, mutate_block]^1 	#将变异点的二进制为反转
    
    return alloChild





if __name__=="__main__":
    '''区块信息 节点信息 约束要求'''
    blockBackups = 3 #区块备份数量 至少为1
    storageLimit = 0.8 #优化后系统总存储空间占比
    #区块信息
    blockInfo = np.loadtxt('./data/blockInfo30.csv', delimiter=',')
    blockNum = blockInfo.shape[0] #区块数量
    #节点信息
    nodeInfo = np.loadtxt('./data/nodeInfo10.csv', delimiter=',')
    nodeNum = nodeInfo.shape[0] #节点数量
    
    #所有节点需要的花销shape=(blockNum,nodeNum,nodeNum)
    costAll = (blockInfo[:,1]*blockInfo[:,2]).reshape((blockNum,1,1)) * nodeInfo[:,2:].reshape((1,nodeNum,nodeNum))
    
    '''遗传算法信息'''
    epochNum = 10 #迭代次数
    bestBlockAllo = np.zeros((nodeNum,blockNum)).astype(bool) #目前最优分配
    alloFitEpoch = np.zeros((epochNum+1,3)) #每轮训练的最优结果 通信成本 存储平衡度 总目标
    
    #原始种群
    alloNum = 200 #种群数量
    #blockAlloGen = generate_blockAllo(alloNum) #生成原始种群
    blockAlloGen = generate_blockAllo_backups(alloNum)
    alloFit = cal_fitness(blockAlloGen) #计算原始种群的 总目标 适应度
    alloFitEpoch[0,:] = alloFit[np.argmax(alloFit[:,3]),1:]
    
    #迭代
    for epoch in range(epochNum):
        blockAlloGen = crossover(blockAlloGen, CROSSOVER_RATE=0.8) #交叉变异
        print(blockAlloGen.shape[0])
        alloFit = cal_fitness(blockAlloGen) #计算种群适应度
        blockAlloGen,alloFit = select(blockAlloGen, alloFit, selectSize=alloNum) #选择个体
        alloFitEpoch[epoch+1,:] = alloFit[np.argmin(alloFit[:,3]),1:] #这一轮的最优值
        bestBlockAllo = blockAlloGen[np.argmin(alloFit[:,3]),:,:] #这一轮的最优分配
    
    #结果
    plt.plot(range(epochNum+1), alloFitEpoch[:,2],'.')
    #print(bestBlockAllo)
    
    
    


